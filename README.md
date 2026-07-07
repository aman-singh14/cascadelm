CascadeLM is a novel method of token cost optimization based on IDK classifier cascades originally implemented with convolutional neural networks to optimize latency. The core idea is built around the following points:

1. Smaller, less "intelligent" models use are less expensive per token
2. Said smaller models can still complete many tasks that heavier models are used for
3. Though people know smaller models are cheaper, they often use heavier models for tasks because they want quality output and confidence in the generated response
4. Thus, the issue is people are spending way more money on responses they could've got cheaper

CascadeLM removes the process of the user having to decide what model to use to save cost while making sure they're getting quality results. 

The process is as follows:

1. CascadeLM does a probe pass with a small model
2. The model calculates the mean entropy of the initial response and if the mean entropy is above a certain threshold (more on this later), then the cascade escalates
3. If the model escalates, the response is regenerated with a larger model. If the model doesn't escalate, the response the user sees is simply the initial response from the small model

This process can be repeated with many models (for now I'm only testing 2) but the cascade relies on the premise that even a few queries can be done with a small model. This is because if the model escalates, the total cost would include both the small model and the heavier models; however, in the case that the model does not escalate, the cost to the user is only of the small model which most times is marginal compared to the cost of a larger model (e.g. gpt 4o mini vs gpt 4o). In the end, it will be cheaper than just using the bigger model for everything. 

It is important to note there a few limitations with this idea:

1. Small models can confidently hallucinate and generate wrong answers when they should've escalated. 
2. Sometimes the smaller model isn't confident about how to format the response rather than the content of the response itself causing unnecessary escalation. 
3. Sometimes the model basically dodges the question and admits it doesn't know but doesn't escalate. This is because the model is confident that it doesn't know the mean entropy of the response isn't high.
4. Subjective questions (e.g. Who is the most underrated philosopher?) escalate but don't necessarily yield any improvements in quality. This is because there are many answers that could work which raises the entropy; however, the same issue applies to heavier models.

Finally, I'll show you how to actually use CascadeLM. 

To install, simply clone my repo from GitHub and run the commands below:

git clone https://github.com/aman-singh14/cascadelm
cd cascadelm
pip install -e .

Basic usage:

from cascadelm import CascadeClient
client = CascadeClient(entropy_threshold=0.4)
response = client.chat(messages=[...])
print(response.content)
print(response.confidence)
client.print_session_summary()

The messages parameter is formatted the same as the OpenAI SDK:

# Text only
response = client.chat(messages=[
    {"role": "user", "content": "What is the capital of France?"}
])

# With system prompt
response = client.chat(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
])

# Multi-turn conversation
response = client.chat(messages=[
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What about Germany?"}
])

# Multimodal — image + text
response = client.chat(messages=[
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://..."}},
            {"type": "text", "text": "What's in this image?"}
        ]
    }
])

Note that the user has the freedom to configure the entropy threshold when initializing CascadeClient. There are tradeoffs of having a high and low threshold. A high threshold is best for minimum cost but may lack in quality for harder queries while a lower threshold would escalate more often if the user wants peace of mind and more confidence in their responses (but more expensive). I will be doing extensive benchmarking to determine optimal entropy thresholds across different domains. 

The confidence object consists of the model used, the mean entropy, and low-confidence spans (groups of tokens were flagged for being uncertain, may be useful to review these areas). Below are examples of the output:

Q: What is the capital of Japan?
Model: gpt-4o-mini | Entropy: 0.0004
Spans: [] 
Q: Who is the most underrated philosopher?
Model: gpt-4o | Entropy: 0.4949
Spans: [{'text': 'Identifying the', 'start_token': 0, 'end_token': 2, 'mean_span_entropy': 0.3926259046543367}, {'text': 'philosopher" can be subjective and depend on personal perspectives and cultural contexts. However', 'start_token': 6, 'end_token': 20, 'mean_span_entropy': 0.5843529211388641}, {'text': 'one philosopher often mentioned in discussions of underrated thinkers is Simone de Beauvoir', 'start_token': 22, 'end_token': 35, 'mean_span_entropy': 0.5083832774817184}, {'text': 'While she is recognized for her contributions to existentialism and feminist theory, her works, particularly "The', 'start_token': 37, 'end_token': 56, 'mean_span_entropy': 0.4232560669075623}, {'text': 'sometimes do not receive as much attention as those of her male contemporaries, like Jean-Paul Sart', 'start_token': 60, 'end_token': 78, 'mean_span_entropy': 0.43317198488482755}, {'text': 'or Martin Heide', 'start_token': 80, 'end_token': 82, 'mean_span_entropy': 0.23839992173305505}, {'text': '.\n\nAnother philosopher that might be considered underrated', 'start_token': 84, 'end_token': 91, 'mean_span_entropy': 0.6912354040822297}, {'text': 'William James, an American pragmatist whose ideas on belief, truth, and the self have significant implications in both philosophy and', 'start_token': 93, 'end_token': 117, 'mean_span_entropy': 0.5516332942922157}, {'text': ', yet he often does not receive as much recognition in mainstream philosophical discourse.\n\nUltimately, the question of who', 'start_token': 119, 'end_token': 139, 'mean_span_entropy': 0.637261021960503}, {'text': 'underrated can vary widely among different audiences, disciplines, and philosophical movements.', 'start_token': 141, 'end_token': 154, 'mean_span_entropy': 0.6669867939481222}]

The session summary comprises of the percentage of queries that were not escalated, the percentage of queries that were escalated, the cost due to the small model, the cost due to escalation, and the hypothetical cost if the user was just using the big model the entire time. Based off that, the savings by using CascadeLM are also provided. Below is an example of the output:

Queries Routed: 3
gpt-4o-mini: 50.00% ($0.00)
gpt-4o: 50.00% ($0.02)
Total Cost: $0.02
Without cascade: $0.03
Savings: $0.01 (37.60%)

Let me know if you have any questions or suggestions. Thank you and enjoy.