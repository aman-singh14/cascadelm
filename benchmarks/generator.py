"""
CascadeLM Benchmark Query Generator
=====================================
Generates realistic agent turn sequences grounded in real source files
from fastapi/full-stack-fastapi-template @ cd83fc1.

HOW IT WORKS
------------
1. REPO_FILES contains the actual source code, hardcoded verbatim.
   This means the benchmark has zero external dependencies — no cloning,
   no network calls, no file paths. Anyone can run it immediately.

2. FILE_COMBOS defines which files an agent would realistically have read
   before its next turn, per phase and complexity. This is the "grounding"
   — the tool results in the messages array contain real code.

3. The generator calls gpt-4o to write the CONVERSATIONAL FRAMING only:
   - a realistic user intent (the opening message)
   - 1-3 prior assistant turns (showing what files the agent read)
   These are the only synthetic parts. The file contents are always real.

4. A routing label is assigned based on phase + complexity:
   - polish/simple      → tier 1 (mini handles fine)
   - debugging/simple   → tier 1
   - debugging/complex  → tier 2 (context is large, bug is subtle)
   - implementation     → tier 2
   - planning           → tier 3 (system design, needs best model)

5. After generation, every query goes through a manual review pass.
   You read each one and delete anything that feels fake or off.

UNDERSTANDING THE MESSAGES ARRAY
---------------------------------
Each query is a `messages` list — the exact format sent to the OpenAI
chat completions API. This is what the proxy will see in production.

   [system]      → tells the model what codebase it's in
   [user]        → the developer's request (synthetic, generated)
   [assistant]   → agent's prior turn: "let me read X" (synthetic)
   [tool]        → file contents from read_file call (REAL)
   [assistant]   → agent's next prior turn if needed (synthetic)
   [tool]        → more file contents (REAL)
   ← routing decision happens here, before the next assistant turn

The entropy signal runs on the model's NEXT completion given this context.
That's what we're testing: does high entropy here predict that mini
will produce a worse answer than gpt-4o?

USAGE
-----
   python benchmarks/generator.py --phase debugging --complexity complex --n 5
   python benchmarks/generator.py --n 30          # generate full corpus
   python benchmarks/generator.py --dry-run       # show prompts only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Real source files from fastapi/full-stack-fastapi-template @ cd83fc1 ─────
# These are hardcoded verbatim so the benchmark has no external dependencies.
# The contents match exactly what was fetched from the repo.

REPO_FILES: dict[str, str] = {

"backend/app/models.py": '''\
import uuid
from datetime import UTC, datetime

from pydantic import EmailStr
from sqlalchemy import DateTime
from sqlmodel import Field, Relationship, SQLModel


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)

class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)

class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)

class UserUpdate(SQLModel):
    email: EmailStr | None = Field(default=None, max_length=255)
    is_active: bool | None = None
    is_superuser: bool | None = None
    full_name: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)

class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)

class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)

class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
    )
    items: list["Item"] = Relationship(back_populates="owner", cascade_delete=True)

class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime | None = None

class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int

class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)

class ItemCreate(ItemBase):
    pass

class ItemUpdate(SQLModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)

class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")

class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime | None = None

class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int

class Message(SQLModel):
    message: str

class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(SQLModel):
    sub: str | None = None

class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)
''',

"backend/app/crud.py": '''\
import uuid
from typing import Any

from sqlmodel import Session, select

from app.core.security import get_password_hash, verify_password
from app.models import Item, ItemCreate, User, UserCreate, UserUpdate


def create_user(*, session: Session, user_create: UserCreate) -> User:
    db_obj = User.model_validate(
        user_create, update={"hashed_password": get_password_hash(user_create.password)}
    )
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def update_user(*, session: Session, db_user: User, user_in: UserUpdate) -> Any:
    user_data = user_in.model_dump(exclude_unset=True)
    extra_data = {}
    if "password" in user_data:
        password = user_data["password"]
        hashed_password = get_password_hash(password)
        extra_data["hashed_password"] = hashed_password
    db_user.sqlmodel_update(user_data, update=extra_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def get_user_by_email(*, session: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    session_user = session.exec(statement).first()
    return session_user


DUMMY_HASH = "$argon2id$v=19$m=65536,t=3,p=4$MjQyZWE1MzBjYjJlZTI0Yw$YTU4NGM5ZTZmYjE2NzZlZjY0ZWY3ZGRkY2U2OWFjNjk"


def authenticate(*, session: Session, email: str, password: str) -> User | None:
    db_user = get_user_by_email(session=session, email=email)
    if not db_user:
        verify_password(password, DUMMY_HASH)
        return None
    verified, updated_password_hash = verify_password(password, db_user.hashed_password)
    if not verified:
        return None
    if updated_password_hash:
        db_user.hashed_password = updated_password_hash
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
    return db_user


def create_item(*, session: Session, item_in: ItemCreate, owner_id: uuid.UUID) -> Item:
    db_item = Item.model_validate(item_in, update={"owner_id": owner_id})
    session.add(db_item)
    session.commit()
    session.refresh(db_item)
    return db_item
''',

"backend/app/api/deps.py": '''\
from collections.abc import Generator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pydantic import ValidationError
from sqlmodel import Session

from app.core import security
from app.core.config import settings
from app.core.db import engine
from app.models import TokenPayload, User

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)

def get_db() -> Generator[Session]:
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_db)]
TokenDep = Annotated[str, Depends(reusable_oauth2)]

def get_current_user(session: SessionDep, token: TokenDep) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (InvalidTokenError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = session.get(User, token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]

def get_current_active_superuser(current_user: CurrentUser) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403, detail="The user doesn\'t have enough privileges"
        )
    return current_user
''',

"backend/app/api/routes/items.py": '''\
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import Item, ItemCreate, ItemPublic, ItemsPublic, ItemUpdate, Message

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/", response_model=ItemsPublic)
def read_items(
    session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100
) -> Any:
    if current_user.is_superuser:
        count_statement = select(func.count()).select_from(Item)
        count = session.exec(count_statement).one()
        statement = (
            select(Item).order_by(col(Item.created_at).desc()).offset(skip).limit(limit)
        )
        items = session.exec(statement).all()
    else:
        count_statement = (
            select(func.count())
            .select_from(Item)
            .where(Item.owner_id == current_user.id)
        )
        count = session.exec(count_statement).one()
        statement = (
            select(Item)
            .where(Item.owner_id == current_user.id)
            .order_by(col(Item.created_at).desc())
            .offset(skip)
            .limit(limit)
        )
        items = session.exec(statement).all()
    items_public = [ItemPublic.model_validate(item) for item in items]
    return ItemsPublic(data=items_public, count=count)

@router.get("/{id}", response_model=ItemPublic)
def read_item(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> Any:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return item

@router.post("/", response_model=ItemPublic)
def create_item(*, session: SessionDep, current_user: CurrentUser, item_in: ItemCreate) -> Any:
    item = Item.model_validate(item_in, update={"owner_id": current_user.id})
    session.add(item)
    session.commit()
    session.refresh(item)
    return item

@router.put("/{id}", response_model=ItemPublic)
def update_item(*, session: SessionDep, current_user: CurrentUser, id: uuid.UUID, item_in: ItemUpdate) -> Any:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    update_dict = item_in.model_dump(exclude_unset=True)
    item.sqlmodel_update(update_dict)
    session.add(item)
    session.commit()
    session.refresh(item)
    return item

@router.delete("/{id}")
def delete_item(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> Message:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    session.delete(item)
    session.commit()
    return Message(message="Item deleted successfully")
''',

"backend/app/api/routes/login.py": '''\
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm

from app import crud
from app.api.deps import CurrentUser, SessionDep, get_current_active_superuser
from app.core import security
from app.core.config import settings
from app.models import Message, NewPassword, Token, UserPublic, UserUpdate
from app.utils import (
    generate_password_reset_token,
    generate_reset_password_email,
    send_email,
    verify_password_reset_token,
)

router = APIRouter(tags=["login"])

@router.post("/login/access-token")
def login_access_token(
    session: SessionDep, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    user = crud.authenticate(
        session=session, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return Token(
        access_token=security.create_access_token(
            user.id, expires_delta=access_token_expires
        )
    )

@router.post("/password-recovery/{email}")
def recover_password(email: str, session: SessionDep) -> Message:
    user = crud.get_user_by_email(session=session, email=email)
    if user:
        password_reset_token = generate_password_reset_token(email=email)
        email_data = generate_reset_password_email(
            email_to=user.email, email=email, token=password_reset_token
        )
        send_email(
            email_to=user.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )
    return Message(
        message="If that email is registered, we sent a password recovery link"
    )

@router.post("/reset-password/")
def reset_password(session: SessionDep, body: NewPassword) -> Message:
    email = verify_password_reset_token(token=body.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid token")
    user = crud.get_user_by_email(session=session, email=email)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid token")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    user_in_update = UserUpdate(password=body.new_password)
    crud.update_user(session=session, db_user=user, user_in=user_in_update)
    return Message(message="Password updated successfully")
''',

"backend/app/api/routes/users.py": '''\
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import col, delete, func, select

from app import crud
from app.api.deps import CurrentUser, SessionDep, get_current_active_superuser
from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.models import (
    Item, Message, UpdatePassword, User, UserCreate,
    UserPublic, UserRegister, UsersPublic, UserUpdate, UserUpdateMe,
)
from app.utils import generate_new_account_email, send_email

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", dependencies=[Depends(get_current_active_superuser)], response_model=UsersPublic)
def read_users(session: SessionDep, skip: int = 0, limit: int = 100) -> Any:
    count_statement = select(func.count()).select_from(User)
    count = session.exec(count_statement).one()
    statement = select(User).order_by(col(User.created_at).desc()).offset(skip).limit(limit)
    users = session.exec(statement).all()
    users_public = [UserPublic.model_validate(user) for user in users]
    return UsersPublic(data=users_public, count=count)

@router.patch("/me/password", response_model=Message)
def update_password_me(*, session: SessionDep, body: UpdatePassword, current_user: CurrentUser) -> Any:
    verified, _ = verify_password(body.current_password, current_user.hashed_password)
    if not verified:
        raise HTTPException(status_code=400, detail="Incorrect password")
    if body.current_password == body.new_password:
        raise HTTPException(status_code=400, detail="New password cannot be the same as the current one")
    hashed_password = get_password_hash(body.new_password)
    current_user.hashed_password = hashed_password
    session.add(current_user)
    session.commit()
    return Message(message="Password updated successfully")

@router.post("/signup", response_model=UserPublic)
def register_user(session: SessionDep, user_in: UserRegister) -> Any:
    user = crud.get_user_by_email(session=session, email=user_in.email)
    if user:
        raise HTTPException(status_code=400, detail="The user with this email already exists in the system")
    user_create = UserCreate.model_validate(user_in)
    user = crud.create_user(session=session, user_create=user_create)
    return user

@router.delete("/{user_id}", dependencies=[Depends(get_current_active_superuser)])
def delete_user(session: SessionDep, current_user: CurrentUser, user_id: uuid.UUID) -> Message:
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user == current_user:
        raise HTTPException(status_code=403, detail="Super users are not allowed to delete themselves")
    statement = delete(Item).where(col(Item.owner_id) == user_id)
    session.exec(statement)
    session.delete(user)
    session.commit()
    return Message(message="User deleted successfully")
''',

"backend/app/core/security.py": '''\
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.core.config import settings

password_hash = PasswordHash(
    (
        Argon2Hasher(),
        BcryptHasher(),
    )
)

ALGORITHM = "HS256"

def create_access_token(subject: str | Any, expires_delta: timedelta) -> str:
    expire = datetime.now(UTC) + expires_delta
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> tuple[bool, str | None]:
    return password_hash.verify_and_update(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return password_hash.hash(password)
''',

"frontend/src/routes/_layout/items.tsx": '''\
import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Search } from "lucide-react"
import { Suspense } from "react"

import { ItemsService } from "@/client"
import { DataTable } from "@/components/Common/DataTable"
import AddItem from "@/components/Items/AddItem"
import { columns } from "@/components/Items/columns"
import PendingItems from "@/components/Pending/PendingItems"

function getItemsQueryOptions() {
  return {
    queryFn: () => ItemsService.readItems({ skip: 0, limit: 100 }),
    queryKey: ["items"],
  }
}

export const Route = createFileRoute("/_layout/items")({
  component: Items,
})

function ItemsTableContent() {
  const { data: items } = useSuspenseQuery(getItemsQueryOptions())
  if (items.data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center py-12">
        <Search className="h-8 w-8 text-muted-foreground" />
        <h3 className="text-lg font-semibold">You don\'t have any items yet</h3>
      </div>
    )
  }
  return <DataTable columns={columns} data={items.data} />
}

function Items() {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Items</h1>
        <AddItem />
      </div>
      <Suspense fallback={<PendingItems />}>
        <ItemsTableContent />
      </Suspense>
    </div>
  )
}
''',

"frontend/src/components/Admin/DeleteUser.tsx": '''\
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Trash2 } from "lucide-react"
import { useState } from "react"
import { useForm } from "react-hook-form"

import { UsersService } from "@/client"
import { Button } from "@/components/ui/button"
import {
  Dialog, DialogClose, DialogContent, DialogDescription,
  DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog"
import { DropdownMenuItem } from "@/components/ui/dropdown-menu"
import { LoadingButton } from "@/components/ui/loading-button"
import useCustomToast from "@/hooks/useCustomToast"
import { handleError } from "@/utils"

interface DeleteUserProps {
  id: string
  onSuccess: () => void
}

const DeleteUser = ({ id, onSuccess }: DeleteUserProps) => {
  const [isOpen, setIsOpen] = useState(false)
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const { handleSubmit } = useForm()

  const mutation = useMutation({
    mutationFn: (id: string) => UsersService.deleteUser({ userId: id }),
    onSuccess: () => {
      showSuccessToast("The user was deleted successfully")
      setIsOpen(false)
      onSuccess()
    },
    onError: handleError.bind(showErrorToast),
    onSettled: () => {
      queryClient.invalidateQueries()
    },
  })

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuItem variant="destructive" onClick={() => setIsOpen(true)}>
        <Trash2 />
        Delete User
      </DropdownMenuItem>
      <DialogContent className="sm:max-w-md">
        <form onSubmit={handleSubmit(() => mutation.mutate(id))}>
          <DialogHeader>
            <DialogTitle>Delete User</DialogTitle>
            <DialogDescription>
              All items associated with this user will also be permanently deleted.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-4">
            <DialogClose asChild>
              <Button variant="outline">Cancel</Button>
            </DialogClose>
            <LoadingButton variant="destructive" type="submit" loading={mutation.isPending}>
              Delete
            </LoadingButton>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

export default DeleteUser
''',
}

# ── File combinations per phase/complexity ────────────────────────────────────
# Which files would a real agent have read before its next turn?
# This is the core grounding decision — derived from how bugs and features
# actually span module boundaries in this codebase.

FILE_COMBOS: dict[str, dict[str, list[str]]] = {
    "polish": {
        # Polish = one file, one function. Agent reads exactly what it needs to touch.
        "simple":   ["backend/app/crud.py"],
        "moderate": ["backend/app/api/routes/items.py"],
        "complex":  ["backend/app/api/routes/users.py"],
    },
    "debugging": {
        # Simple bugs are self-contained in one or two files.
        "simple":   ["backend/app/api/routes/items.py"],
        # Moderate bugs cross one boundary (route → crud, or route → model).
        "moderate": ["backend/app/api/routes/users.py", "backend/app/crud.py"],
        # Complex bugs cross multiple layers — the verify_password tuple issue
        # is the canonical example: crud.py calls security.py wrong.
        "complex":  ["backend/app/crud.py", "backend/app/core/security.py",
                     "backend/app/api/routes/users.py"],
    },
    "implementation": {
        # Implementation always needs the pattern files: models + the relevant route.
        "simple":   ["backend/app/models.py", "backend/app/api/routes/items.py"],
        "moderate": ["backend/app/models.py", "backend/app/crud.py",
                     "backend/app/api/routes/items.py"],
        # Complex implementation spans frontend too.
        "complex":  ["backend/app/models.py", "backend/app/crud.py",
                     "backend/app/api/routes/items.py",
                     "frontend/src/routes/_layout/items.tsx"],
    },
    "planning": {
        # Planning is mostly user intent — agent may have read models to understand
        # the data shape, but hasn't dug into routes yet.
        "simple":   ["backend/app/models.py"],
        "moderate": ["backend/app/models.py", "backend/app/api/routes/items.py"],
        "complex":  ["backend/app/models.py", "backend/app/crud.py",
                     "backend/app/api/routes/items.py",
                     "backend/app/api/routes/users.py"],
    },
}

# ── Routing labels ────────────────────────────────────────────────────────────
# Tier 1 = mini, Tier 2 = gpt-4o, Tier 3 = o3/opus
# Rationale:
#   polish/simple    → mini: single function, low ambiguity
#   debugging/simple → mini: contained bug, answer is obvious from context
#   debugging/moderate→ tier2: crosses a boundary, subtle
#   debugging/complex→ tier2: multi-file, easy to hallucinate confidently
#   implementation/* → tier2/3: novel code generation
#   planning/*       → tier3: system design needs best reasoning

ROUTING_LABELS: dict[str, dict[str, int]] = {
    "polish":         {"simple": 1, "moderate": 1, "complex": 2},
    "debugging":      {"simple": 1, "moderate": 2, "complex": 2},
    "implementation": {"simple": 2, "moderate": 2, "complex": 3},
    "planning":       {"simple": 2, "moderate": 3, "complex": 3},
}

SYSTEM_PROMPT = (
    "You are a coding agent working inside fastapi/full-stack-fastapi-template "
    "(commit cd83fc1, MIT licence). You have access to read_file and edit_file tools. "
    "The stack is: FastAPI + SQLModel + PostgreSQL backend, "
    "React + TypeScript + TanStack Query frontend, Docker Compose deployment."
)

# ── Scenario seeds ────────────────────────────────────────────────────────────
# Each seed pins the TOPIC for a generated query so we get real diversity across
# the corpus. Without seeds, gpt-4o gravitates to the most obvious scenario for
# a file combination and produces paraphrases of the same query.
#
# Seeds cycle: if we need 30 queries from 6 seeds, each seed appears ~5 times
# but with different phrasing (gpt-4o varies the wording).
#
# All identifiers in "focus" are exact strings from REPO_FILES — verified by
# reading the actual files. Never add an identifier you haven't confirmed.

SCENARIO_SEEDS: dict[tuple[str, str], list[dict]] = {

    ("polish", "simple"): [
        # file: backend/app/crud.py
        {
            "scenario": "update_user return type is annotated as Any — change it to the specific return type User",
            "focus": ["update_user", "Any", "User", "session.refresh"],
        },
        {
            "scenario": "DUMMY_HASH has no comment explaining why it exists — add a concise inline comment",
            "focus": ["DUMMY_HASH", "authenticate", "verify_password"],
        },
        {
            "scenario": "get_user_by_email stores the query result in an intermediate variable session_user then immediately returns it — inline the return",
            "focus": ["get_user_by_email", "session_user", "statement", "session.exec"],
        },
        {
            "scenario": "create_user and create_item both repeat the same session.add / session.commit / session.refresh pattern — add a comment noting the shared pattern",
            "focus": ["create_user", "create_item", "session.add", "session.commit"],
        },
        {
            "scenario": "authenticate has no explicit return type annotation — add -> User | None",
            "focus": ["authenticate", "User", "get_user_by_email"],
        },
        {
            "scenario": "update_user builds extra_data as an empty dict then conditionally populates it — simplify with a direct dict literal in the common path",
            "focus": ["update_user", "extra_data", "user_data", "sqlmodel_update"],
        },
    ],

    ("polish", "moderate"): [
        # file: backend/app/api/routes/items.py
        {
            "scenario": "delete_item returns Message(message='Item deleted successfully') but the route has no explicit status_code — add status_code=200 to make the intent explicit",
            "focus": ["delete_item", "Message", "session.delete", "session.commit"],
        },
        {
            "scenario": "read_items uses the magic number 100 as the default limit — extract it to a named constant at the module level",
            "focus": ["read_items", "limit", "offset", "skip"],
        },
        {
            "scenario": "read_item and delete_item both repeat the same ownership check (item.owner_id != current_user.id) — add a comment noting they should stay in sync",
            "focus": ["read_item", "delete_item", "owner_id", "is_superuser"],
        },
        {
            "scenario": "update_item calls model_dump(exclude_unset=True) without a comment explaining why exclude_unset is required — add a clarifying comment",
            "focus": ["update_item", "model_dump", "exclude_unset", "sqlmodel_update"],
        },
        {
            "scenario": "create_item in routes shadows the imported crud.create_item — rename the route function to avoid the collision",
            "focus": ["create_item", "ItemCreate", "owner_id", "current_user"],
        },
        {
            "scenario": "read_items count_statement duplicated in superuser and non-superuser branches — the superuser branch could reuse select(func.count())",
            "focus": ["read_items", "count_statement", "func.count", "select"],
        },
    ],

    ("debugging", "simple"): [
        # file: backend/app/api/routes/items.py
        {
            "scenario": "developer removed session.refresh(item) from update_item thinking it was redundant — now the endpoint returns the pre-update values",
            "symptom": "PUT /items/{id} returns the old title even though the database was updated",
            "focus": ["update_item", "session.refresh", "sqlmodel_update", "item_in"],
        },
        {
            "scenario": "read_items returns an empty list when limit=0 is passed — no error, just silence",
            "symptom": "GET /items?limit=0 returns {data: [], count: N} instead of an error",
            "focus": ["read_items", "limit", "offset", "skip"],
        },
        {
            "scenario": "delete_item returns 403 for items that don't belong to the user even when the item doesn't exist — leaks item existence to non-owners",
            "symptom": "non-superuser gets 403 on a random UUID instead of 404",
            "focus": ["delete_item", "is_superuser", "HTTPException", "owner_id"],
        },
        {
            "scenario": "create_item owner_id is set from current_user.id but developer expects to pass it in the request body via ItemCreate",
            "symptom": "POST /items always sets owner to the authenticated user regardless of body content",
            "focus": ["create_item", "model_validate", "owner_id", "current_user"],
        },
        {
            "scenario": "read_items returns items in descending creation order but developer expected ascending — order_by uses .desc()",
            "symptom": "newest items appear first; developer wants oldest first",
            "focus": ["read_items", "order_by", "created_at", "col"],
        },
        {
            "scenario": "read_item endpoint returns a raw Item (table model) instead of ItemPublic — internal fields may be exposed",
            "symptom": "GET /items/{id} response includes fields not present in ItemPublic",
            "focus": ["read_item", "ItemPublic", "response_model", "session.get"],
        },
        {
            "scenario": "non-superuser calls read_items and gets items from other users because developer added the superuser branch items query without the WHERE clause",
            "symptom": "regular user can see all items in the system",
            "focus": ["read_items", "is_superuser", "owner_id", "CurrentUser"],
        },
    ],

    ("debugging", "moderate"): [
        # files: backend/app/api/routes/users.py + backend/app/crud.py
        {
            "scenario": "update_password_me discards the second return value of verify_password with _ — if pwdlib needs to upgrade the argon2 hash, the new hash is never saved",
            "symptom": "after upgrading pwdlib, some users' hashes never migrate even after password change",
            "focus": ["update_password_me", "verify_password", "hashed_password", "get_password_hash"],
        },
        {
            "scenario": "developer removed the verify_password(password, DUMMY_HASH) call in authenticate's user-not-found branch thinking it was dead code — response time now differs for valid vs invalid emails",
            "symptom": "timing side-channel allows enumeration of registered emails",
            "focus": ["authenticate", "DUMMY_HASH", "verify_password", "get_user_by_email"],
        },
        {
            "scenario": "delete_user does not prevent deleting other superusers — the guard only checks user == current_user",
            "symptom": "superuser A can delete superuser B with no error",
            "focus": ["delete_user", "is_superuser", "current_user", "HTTPException"],
        },
        {
            "scenario": "register_user checks get_user_by_email then calls create_user — not atomic, so two simultaneous registrations with the same email both pass the check and the second commit raises an unhandled error",
            "symptom": "occasional 500 on /signup under load when two users register with the same email at the same time",
            "focus": ["register_user", "get_user_by_email", "create_user", "session.commit"],
        },
        {
            "scenario": "update_user in crud.py re-hashes the password when it appears in user_data — if caller passes an already-hashed value by mistake, it gets double-hashed and the account is locked",
            "symptom": "user can no longer log in after an admin update that included hashed_password in the request",
            "focus": ["update_user", "get_password_hash", "hashed_password", "user_data"],
        },
        {
            "scenario": "get_user_by_email is case-sensitive — User@Example.com and user@example.com register as separate accounts despite EmailStr normalizing on input",
            "symptom": "duplicate accounts for the same email with different casing",
            "focus": ["get_user_by_email", "email", "select", "User.email"],
        },
    ],

    ("debugging", "complex"): [
        # files: backend/app/crud.py + backend/app/core/security.py + backend/app/api/routes/users.py
        {
            "scenario": "verify_password returns tuple[bool, str | None] — if developer writes `if not verify_password(...)` instead of unpacking, the tuple is always truthy so `not tuple` is always False and any password is accepted",
            "symptom": "password change endpoint accepts any string as the current_password",
            "focus": ["verify_password", "update_password_me", "verified", "tuple[bool, str | None]"],
        },
        {
            "scenario": "authenticate persists hash upgrades (db_user.hashed_password = updated_password_hash) but update_password_me discards them with _ — hashes upgrade on login but not after password change",
            "symptom": "after migrating from bcrypt to argon2, user hashes only upgrade if they log in, not if they change their password",
            "focus": ["authenticate", "updated_password_hash", "update_password_me", "hashed_password"],
        },
        {
            "scenario": "developer removed verify_password(password, DUMMY_HASH) from authenticate's user-not-found path — the branch that prevented timing attacks is gone",
            "symptom": "response latency is measurably shorter for invalid emails than valid ones",
            "focus": ["authenticate", "DUMMY_HASH", "verify_password", "get_user_by_email"],
        },
        {
            "scenario": "delete_user uses session.exec(delete(Item)...) explicitly but the Item relationship already has cascade_delete=True and ondelete=CASCADE — double deletion",
            "symptom": "delete_user occasionally raises an error in staging with certain DB isolation levels",
            "focus": ["delete_user", "cascade_delete", "delete", "session.exec"],
        },
        {
            "scenario": "UserUpdateMe is imported in users.py but no route uses it — there is no PATCH /users/me endpoint for updating display name or email, only PATCH /users/me/password exists",
            "symptom": "frontend cannot update the user's full_name — no endpoint exists despite the model being defined",
            "focus": ["UserUpdateMe", "update_password_me", "full_name", "UserUpdate"],
        },
        {
            "scenario": "register_user and get_user_by_email together are not atomic — under concurrent load two requests pass the duplicate-email check and the second session.commit raises an unhandled IntegrityError",
            "symptom": "intermittent 500 on POST /users/signup under load testing",
            "focus": ["register_user", "get_user_by_email", "create_user", "session.commit"],
        },
    ],

    ("implementation", "simple"): [
        # files: backend/app/models.py + backend/app/api/routes/items.py
        {
            "scenario": "add a status field to Item (values: active, archived) and filter read_items to exclude archived by default",
            "focus": ["Item", "ItemCreate", "ItemUpdate", "read_items"],
        },
        {
            "scenario": "add a due_date: datetime | None field to ItemBase so it flows through to ItemCreate, ItemUpdate, and ItemPublic",
            "focus": ["ItemBase", "Item", "ItemCreate", "ItemPublic"],
        },
        {
            "scenario": "add a GET /items/{id}/duplicate endpoint that creates a copy of an item for the current user",
            "focus": ["create_item", "ItemPublic", "read_item", "router"],
        },
        {
            "scenario": "add a title query parameter to read_items for substring filtering — return only items whose title contains the query string",
            "focus": ["read_items", "ItemsPublic", "select", "func.count"],
        },
        {
            "scenario": "add a is_pinned: bool field to ItemBase and sort pinned items first in read_items",
            "focus": ["Item", "ItemBase", "ItemPublic", "read_items"],
        },
        {
            "scenario": "add a GET /items/count endpoint that returns just the integer count of items for the authenticated user",
            "focus": ["read_items", "func.count", "router", "CurrentUser"],
        },
    ],

    ("implementation", "moderate"): [
        # files: backend/app/models.py + backend/app/crud.py + backend/app/api/routes/items.py
        {
            "scenario": "add soft delete: add deleted_at: datetime | None to Item, filter it out of read_items by default, add a DELETE that sets deleted_at instead of removing the row, add a restore endpoint",
            "focus": ["Item", "create_item", "read_items", "session.exec"],
        },
        {
            "scenario": "add a bulk delete endpoint DELETE /items that accepts a list of item IDs and deletes all items owned by the current user in that list",
            "focus": ["delete_item", "Item", "session.exec", "router"],
        },
        {
            "scenario": "add a get_items_by_title crud function and wire it to a ?q= query param in read_items",
            "focus": ["create_item", "select", "Item", "read_items"],
        },
        {
            "scenario": "add a duplicate_item crud function and a POST /items/{id}/duplicate route that copies title and description to a new item owned by the requester",
            "focus": ["create_item", "ItemCreate", "read_item", "router"],
        },
        {
            "scenario": "add an updated_at: datetime field to Item that is set on creation and updated on every PUT — follow the same pattern as created_at",
            "focus": ["Item", "ItemBase", "update_item", "sqlmodel_update"],
        },
        {
            "scenario": "add a GET /items/stats endpoint returning count of active items grouped by owner — superuser only",
            "focus": ["read_items", "func.count", "Item", "select"],
        },
    ],

    ("implementation", "complex"): [
        # files: backend/app/models.py + backend/app/crud.py + backend/app/api/routes/items.py + frontend/src/routes/_layout/items.tsx
        {
            "scenario": "add server-side sorting to read_items (sort_by=title|created_at, order=asc|desc query params) and add sort controls to the Items page in items.tsx",
            "focus": ["read_items", "order_by", "created_at", "useSuspenseQuery"],
        },
        {
            "scenario": "add a client-side title filter to ItemsTableContent — a text input that filters the already-fetched items.data array without a new API call",
            "focus": ["ItemsTableContent", "useSuspenseQuery", "DataTable", "items.data"],
        },
        {
            "scenario": "add a CSV export button to the Items page — a GET /items/export endpoint that returns CSV and a download trigger in items.tsx",
            "focus": ["Items", "read_items", "ItemPublic", "router"],
        },
        {
            "scenario": "add is_pinned to Item model, pin/unpin endpoints in routes/items.py, and render pinned items with a pin icon at the top of the DataTable in items.tsx",
            "focus": ["Item", "ItemBase", "read_items", "ItemsTableContent"],
        },
        {
            "scenario": "switch items.tsx from fetching all 100 items at once to page-based loading — update getItemsQueryOptions to accept page number and pass skip/limit",
            "focus": ["getItemsQueryOptions", "readItems", "skip", "limit"],
        },
    ],

    ("planning", "moderate"): [
        # files: backend/app/models.py + backend/app/api/routes/items.py
        {
            "scenario": "design an audit log system that records who created, updated, or deleted each Item and when",
            "focus": ["Item", "create_item", "update_item", "delete_item"],
        },
        {
            "scenario": "design a notification system that triggers when an item is created or updated — what model changes are needed and how should delivery work",
            "focus": ["Item", "ItemPublic", "router", "owner_id"],
        },
        {
            "scenario": "design a rate limiting strategy for the items API — which endpoints need limits, where should state live, what happens on breach",
            "focus": ["read_items", "create_item", "router", "CurrentUser"],
        },
        {
            "scenario": "design a versioning system for Item — keep a history of every change so users can see what a title or description looked like at any past point",
            "focus": ["Item", "ItemUpdate", "ItemBase", "update_item"],
        },
        {
            "scenario": "design a caching layer for read_items — what to cache, how to invalidate on create/update/delete, whether to cache per-user or globally",
            "focus": ["read_items", "func.count", "select", "ItemsPublic"],
        },
    ],

    ("planning", "complex"): [
        # files: backend/app/models.py + backend/app/crud.py + backend/app/api/routes/items.py + backend/app/api/routes/users.py
        {
            "scenario": "design role-based access control to replace the boolean is_superuser — what roles are needed, how do they map to the existing endpoints, what model changes are required",
            "focus": ["User", "is_superuser", "get_current_active_superuser", "read_users"],
        },
        {
            "scenario": "design a multi-tenant workspace model — items belong to a workspace, users can be in multiple workspaces, how does ownership change",
            "focus": ["Item", "owner_id", "User", "create_item"],
        },
        {
            "scenario": "design item sharing between users — a user can share read-only or read-write access to an item with another user",
            "focus": ["Item", "owner_id", "UserPublic", "ItemPublic"],
        },
        {
            "scenario": "design a subscription tier system — free users can have N items, pro users unlimited, enforce the limit at create time",
            "focus": ["User", "Item", "create_item", "read_items"],
        },
        {
            "scenario": "design a data retention policy — items older than a configurable threshold are auto-archived, users can opt out, superusers can override",
            "focus": ["Item", "created_at", "ItemUpdate", "User"],
        },
    ],
}


# ── Generator prompt ──────────────────────────────────────────────────────────

def build_generator_prompt(
    phase: str,
    complexity: str,
    files_read: list[str],
    seed: dict,
) -> str:
    """
    Builds the prompt sent to gpt-4o to generate the conversational framing.

    Key design decisions:
    - Full file contents are included so gpt-4o can pick real identifiers.
    - The scenario is prescribed by the seed so every call produces a distinct topic.
    - ground_truth_keywords must appear verbatim in the file contents — no guessing.
    """
    # Embed actual file contents so gpt-4o never has to guess what's in them
    file_blocks = "\n\n".join(
        f'<file path="{f}">\n{REPO_FILES[f]}\n</file>' for f in files_read
    )

    phase_guidance = {
        "polish":         "Small cleanup: fix a type annotation, rename a variable, inline a redundant intermediate, add a clarifying comment. NOT a new feature. NOT a bug fix.",
        "debugging":      "Something is broken. Developer has a concrete symptom. Agent needs to find the root cause in the code and fix it.",
        "implementation": "Add a new feature or endpoint that does not exist yet. Agent follows existing patterns.",
        "planning":       "Design a system or architecture. Multiple valid approaches. Tradeoffs must be discussed before any code is written.",
    }[phase]

    symptom_line = (
        f'\nSYMPTOM THE DEVELOPER OBSERVES:\n{seed["symptom"]}\n'
        if "symptom" in seed else ""
    )

    tier = ROUTING_LABELS[phase][complexity]

    return f"""You are generating one query for a benchmark that tests LLM routing decisions.

PHASE: {phase} — {phase_guidance}
COMPLEXITY: {complexity}
CODEBASE: fastapi/full-stack-fastapi-template (FastAPI + SQLModel + PostgreSQL, React + TypeScript)

SCENARIO (use this exact scenario — do not invent a different one):
{seed["scenario"]}
{symptom_line}
KEY IDENTIFIERS IN THIS SCENARIO (all appear verbatim in the files below):
{', '.join(seed['focus'])}

THE AGENT HAS ALREADY READ THESE FILES:
{file_blocks}

---

Generate a JSON object with exactly these fields:

{{
  "user_message": "...",
  "prior_turns": [
    {{"before_file": "path/to/file", "assistant_says": "..."}},
    ...
  ],
  "routing_rationale": "...",
  "ground_truth_keywords": ["...", "...", "..."]
}}

RULES — read carefully:

user_message
  - 1-2 sentences in natural developer tone.
  - Must be consistent with the scenario and symptom above.
  - Never mention file names or say "in this codebase".
  - The scenario is fixed but phrase it differently each time — imagine different
    developers describing the same problem in their own words.
  - Good: "Password change is accepting any current password — it's not checking correctly."
  - Bad: "In this FastAPI template the update_password_me function has a bug."

prior_turns
  - One entry per file listed above, in order.
  - assistant_says: one sentence in first person explaining WHY the agent reads that file.
    Reference what the agent was looking for, not just "let me read X".
  - Good: "I want to check how verify_password is called to see if the return value is handled correctly."
  - Bad: "Let me read backend/app/core/security.py."

routing_rationale
  - 1-2 sentences: why this turn needs tier {tier} and what a cheaper/pricier model
    would likely get wrong.

ground_truth_keywords
  - Exactly 3-5 strings.
  - EVERY keyword must appear verbatim somewhere in the file contents shown above.
    Copy-paste from the code — do not paraphrase, do not invent names.
  - Wrong: "hash_password" (function is get_password_hash), "get_user_by_id" (doesn't exist)
  - Right: "get_password_hash", "verify_password", "updated_password_hash"

Respond with ONLY the JSON object. No markdown, no explanation, no code fences."""


# ── Core generation function ──────────────────────────────────────────────────

def generate_query(
    phase: str,
    complexity: str,
    query_id: str,
    client: OpenAI,
    seed_index: int = 0,
    dry_run: bool = False,
) -> dict:
    """
    Generates one benchmark query.

    Steps:
    1. Look up which files to inject for this phase/complexity
    2. Pick the scenario seed (cycles through SCENARIO_SEEDS for this combo)
    3. Build the generator prompt with full file contents + seed
    4. Call gpt-4o to get user_message + prior_turns (the synthetic framing)
    5. Assemble the full messages array with real file contents in tool results
    6. Return the complete query dict
    """
    files_to_read = FILE_COMBOS[phase][complexity]
    expected_tier  = ROUTING_LABELS[phase][complexity]

    seeds = SCENARIO_SEEDS.get((phase, complexity), [{"scenario": f"{phase}/{complexity} task", "focus": []}])
    seed  = seeds[seed_index % len(seeds)]

    prompt = build_generator_prompt(phase, complexity, files_to_read, seed)

    if dry_run:
        print(f"\n[DRY RUN] {query_id} | {phase}/{complexity} -> tier {expected_tier}")
        print(f"  Seed [{seed_index % len(seeds)}]: {seed['scenario'][:80]}...")
        print(f"  Files: {files_to_read}")
        print(f"  Prompt length: {len(prompt)} chars")
        return {"id": query_id, "dry_run": True, "seed": seed["scenario"]}

    # Call gpt-4o to generate the conversational framing
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.8,   # some variety so queries aren't identical
    )

    raw = response.choices[0].message.content or ""

    # Parse the JSON — strip any accidental markdown fences
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        framing = json.loads(raw[start:end])
    except Exception as e:
        print(f"  WARNING: JSON parse failed for {query_id}: {e}")
        print(f"  Raw: {raw[:200]}")
        return {"id": query_id, "error": str(e), "raw": raw}

    # Assemble the full messages array
    # Structure:
    #   [system] → who the agent is
    #   [user]   → developer's request (synthetic)
    #   for each file:
    #     [assistant] → "let me read X" (synthetic)
    #     [tool]      → actual file contents (REAL)
    # ← routing decision happens after the last tool result

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": framing["user_message"]},
    ]

    prior_turns = framing.get("prior_turns", [])
    for i, file_path in enumerate(files_to_read):
        if i < len(prior_turns):
            assistant_says = prior_turns[i].get("assistant_says", f"Let me read {file_path}.")
        else:
            assistant_says = f"Let me read {file_path} to understand the relevant code."

        tool_call_id = f"call_read_{i}"

        # OpenAI requires: assistant message with tool_calls → tool message with
        # matching tool_call_id. Plain content-only assistant messages followed by
        # role:tool messages are rejected with a 400.
        messages.append({
            "role": "assistant",
            "content": assistant_says,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": file_path}),
                },
            }],
        })

        file_contents = REPO_FILES.get(file_path, f"# File not found: {file_path}")
        messages.append({
            "role": "tool",
            "content": f"<file path='{file_path}'>\n{file_contents}\n</file>",
            "tool_call_id": tool_call_id,
        })

    return {
        "id":                  query_id,
        "phase":               phase,
        "complexity":          complexity,
        "expected_tier":       expected_tier,
        "seed_scenario":       seed["scenario"],
        "files_read":          files_to_read,
        "messages":            messages,
        "ground_truth":        framing.get("ground_truth_keywords", []),
        "routing_rationale":   framing.get("routing_rationale", ""),
        "needs_review":        True,
        "reviewed":            False,
        "keep":                None,
    }


# ── Batch generation ──────────────────────────────────────────────────────────

PHASE_COMPLEXITY_TARGETS = {
    # How many queries to generate per (phase, complexity) combination.
    # Total = 170 queries. Weighted toward debugging because that's
    # where the routing signal is hardest to get right.
    ("polish",         "simple"):   20,
    ("polish",         "moderate"): 10,
    ("debugging",      "simple"):   25,
    ("debugging",      "moderate"): 25,
    ("debugging",      "complex"):  30,
    ("implementation", "simple"):   15,
    ("implementation", "moderate"): 20,
    ("implementation", "complex"):  10,
    ("planning",       "moderate"): 10,
    ("planning",       "complex"):   5,
}


def run_generation(
    target_phase: str | None = None,
    target_complexity: str | None = None,
    n_override: int | None = None,
    dry_run: bool = False,
    sleep_between: float = 0.5,
) -> list[dict]:
    """
    Runs the generator for all (phase, complexity) combinations,
    or just one if target_phase/complexity are specified.
    """
    client = None if dry_run else OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    outfile = output_dir / f"corpus_{timestamp}.json"

    all_queries: list[dict] = []
    counter = 0

    for (phase, complexity), n in PHASE_COMPLEXITY_TARGETS.items():
        # Filter if specific phase/complexity requested
        if target_phase and phase != target_phase:
            continue
        if target_complexity and complexity != target_complexity:
            continue
        if n_override is not None:
            n = n_override

        print(f"\n-- {phase}/{complexity} -> {n} queries (tier {ROUTING_LABELS[phase][complexity]}) --")

        for i in range(n):
            query_id = f"{phase[:3]}_{complexity[:3]}_{counter:03d}"
            counter += 1

            query = generate_query(phase, complexity, query_id, client, seed_index=i, dry_run=dry_run)
            all_queries.append(query)

            if not dry_run:
                print(f"  [{query_id}] ok  seed={i % len(SCENARIO_SEEDS.get((phase, complexity), [{}]))}")
                time.sleep(sleep_between)

    if not dry_run:
        outfile.write_text(json.dumps(all_queries, indent=2))
        print(f"\n{len(all_queries)} queries saved -> {outfile}")
        print(f"\nNext step: run the review pass")
        print(f"  python benchmarks/review.py {outfile}")

    return all_queries


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CascadeLM query generator")
    parser.add_argument("--phase",       choices=["polish","debugging","implementation","planning"])
    parser.add_argument("--complexity",  choices=["simple","moderate","complex"])
    parser.add_argument("--n",           type=int, help="Override query count per combination")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--sleep",       type=float, default=0.5)
    args = parser.parse_args()

    run_generation(
        target_phase=args.phase,
        target_complexity=args.complexity,
        n_override=args.n,
        dry_run=args.dry_run,
        sleep_between=args.sleep,
    )


if __name__ == "__main__":
    main()