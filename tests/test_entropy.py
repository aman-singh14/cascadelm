from cascadelm import CascadeClient, CascadeResponse, ConfidenceMetadata
from cascadelm.entropy import token_entropy, mean_entropy

def test_token_entropy_certain():
    # one token with near-certain probability
    top_logprobs = [("Paris", -0.02), ("Lyon", -5.1)]
    h = token_entropy(top_logprobs)
    assert h < 0.1, f"Expected low entropy, got {h}"

def test_token_entropy_uncertain():
    # evenly spread probabilities
    top_logprobs = [("Hume", -1.2), ("Peirce", -1.4), ("Schopenhauer", -1.9)]
    h = token_entropy(top_logprobs)
    assert h > 0.5, f"Expected high entropy, got {h}"

def test_mean_entropy_empty():
    from cascadelm.entropy import mean_entropy
    assert mean_entropy([]) == 0.0

if __name__ == "__main__":
    test_token_entropy_certain()
    test_token_entropy_uncertain()
    test_mean_entropy_empty()
    print("All tests passed.")