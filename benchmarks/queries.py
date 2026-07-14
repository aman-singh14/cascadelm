queries = [
    # BOILERPLATE — should NOT escalate
    {
        "query": """Here is a Next.js API route that sends contact form emails for a nonprofit website:

```typescript
import { NextRequest, NextResponse } from 'next/server';
import nodemailer from 'nodemailer';

export async function POST(request: NextRequest) {
  try {
    const { name, email, subject, message } = await request.json();
    if (!name || !email || !subject || !message) {
      return NextResponse.json({ error: 'All fields are required' }, { status: 400 });
    }
    const transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST || 'smtp.gmail.com',
      port: parseInt(process.env.SMTP_PORT || '587'),
      secure: false,
      auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
    });
    const mailOptions = {
      from: process.env.SMTP_USER,
      to: 'contact@educationforourfutures.com',
      subject: `Contact Form: ${subject}`,
      html: `<div>${message.replace(/\n/g, '<br>')}</div>`,
      replyTo: email,
    };
    await transporter.sendMail(mailOptions);
    return NextResponse.json({ message: 'Email sent successfully' }, { status: 200 });
  } catch (error) {
    console.error('Error sending email:', error);
    return NextResponse.json({ error: 'Failed to send email' }, { status: 500 });
  }
}
```

Add a rate limiter to this route that allows a maximum of 5 requests per IP address per hour.""",
        "category": "boilerplate",
        "expected_escalation": False
    },
    {
        "query": """Here is a Flask API for checking fishing regulations:

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/check_location', methods=['POST'])
def check_location():
    data = request.json
    region = data.get('region')
    location = data.get('location')
    if not region or not location:
        return jsonify({'error': 'Region and location are required'}), 400
    result = check_fishing_location(region, location)
    return jsonify(result)
```

Add a /health endpoint and a /regulations endpoint that accepts a GET request with query parameters instead of POST with JSON body.""",
        "category": "boilerplate",
        "expected_escalation": False
    },

    # SYSTEM DESIGN — should escalate
    {
        "query": """I have a 3D Gaussian Splatting evaluation pipeline (eval_metrics.py) that currently runs inference sequentially across all images in a COLMAP reconstruction:

```python
for image_id, image in reconstruction.images.items():
    render, alphas, meta = rasterization(
        means=ckpt['splats']['means'],
        scales=torch.exp(ckpt['splats']['scales']),
        opacities=torch.sigmoid(ckpt['splats']['opacities']),
        quats=ckpt['splats']['quats'],
        colors=sh_colors, sh_degree=3,
        viewmats=viewmat_tensor, Ks=K_tensor,
        width=camera.width, height=camera.height
    )
    metrics = compute_metrics(render, gt_image)
```

The scene has 200+ images and evaluation takes ~45 minutes on an RTX 3060. Design a batched evaluation pipeline that parallelizes rendering across multiple viewpoints while keeping GPU memory under 10GB.""",
        "category": "system_design",
        "expected_escalation": True
    },
    {
        "query": """I have a fish classification CNN using ResNet50V2 as backbone:

```python
base_model = tf.keras.applications.ResNet50V2(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
base_model.trainable = False
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

I want to deploy this as a REST API that can handle 50 concurrent classification requests with under 200ms latency. The model file is 200MB. Design the serving infrastructure.""",
        "category": "system_design",
        "expected_escalation": True
    },

    # DEBUGGING — simple, should NOT escalate
    {
        "query": """This eval_metrics.py script crashes with the following error when I run it: TypeError: 
        'ArgumentParser' object is not subscriptable
        Here is the relevant code:

```python
parser = Parser(
    data_dir="/home/aman/gs_project/plant",
    factor=1,
    normalize=True,
    test_every=8
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
```

What is causing this error and how do I fix it?""",
        "category": "debugging",
        "expected_escalation": False
    },
    {
        "query": """My fish CNN training loop runs without errors but validation accuracy is stuck at 9-10% across all epochs regardless of learning rate. Here is the model and training setup:

```python
model = models.Sequential([
    base_model,  # ResNet50V2, frozen
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    data_augmentation(X_train), y_train,
    validation_data=(X_val, y_val),
    epochs=20, batch_size=32
)
```

The dataset has 10 classes with roughly equal distribution. What are the most likely causes and how would you systematically debug this?""",
        "category": "debugging",
        "expected_escalation": True
    },

    # SECURITY — should escalate
    {
        "query": """Here is a Next.js contact form API route used in a nonprofit's public website:

```typescript
export async function POST(request: NextRequest) {
  const { name, email, subject, message } = await request.json();
  if (!name || !email || !subject || !message) {
    return NextResponse.json({ error: 'All fields are required' }, { status: 400 });
  }
  const mailOptions = {
    from: process.env.SMTP_USER,
    to: 'contact@educationforourfutures.com',
    subject: `Contact Form: ${subject}`,
    html: `<div style="...">${message.replace(/\n/g, '<br>')}</div>`,
    replyTo: email,
  };
  await transporter.sendMail(mailOptions);
}
```

Identify all security vulnerabilities in this code and provide fixed implementations for each one.""",
        "category": "security",
        "expected_escalation": True
    },
    {
        "query": """My Flask fishing regulations API is going to production next week. Here is the full route handler:

```python
@app.route('/check_location', methods=['POST'])
def check_location():
    data = request.json
    region = data.get('region')
    location = data.get('location')
    if not region or not location:
        return jsonify({'error': 'Region and location are required'}), 400
    result = check_fishing_location(region, location)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

What security issues exist here and what would a production-hardened version look like?""",
        "category": "security",
        "expected_escalation": True
    },

    # TESTING — should NOT escalate
    {
        "query": """Here is a compute_metrics function from a 3DGS evaluation pipeline:

```python
def compute_metrics(renders, gt_images):
    psnr = PeakSignalNoiseRatio(data_range=2).to("cuda")
    ssim = StructuralSimilarityIndexMeasure(data_range=2).to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity().to("cuda")
    results = {
        "psnr": psnr(renders, gt_images).item(),
        "ssim": ssim(renders, gt_images).item(),
        "lpips": lpips(renders, gt_images).item()
    }
    return results
```

Write pytest unit tests for this function. Include tests for identical images, completely different images, and mismatched tensor shapes.""",
        "category": "testing",
        "expected_escalation": False
    },
    {
        "query": """Here is a Flask API endpoint for checking fishing regulations:

```python
@app.route('/check_location', methods=['POST'])
def check_location():
    data = request.json
    region = data.get('region')
    location = data.get('location')
    if not region or not location:
        return jsonify({'error': 'Region and location are required'}), 400
    result = check_fishing_location(region, location)
    return jsonify(result)
```

Write pytest tests using Flask's test client that cover: missing fields, valid request with mocked check_fishing_location, and invalid JSON body.""",
        "category": "testing",
        "expected_escalation": False
    },
]