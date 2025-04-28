from vllm import LLM, SamplingParams
 
from functools import partial
 
#llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
#llm = LLM(model="deepseek-ai/DeepSeek-V3",
#llm = LLM(model="Qwen/Qwen2.5-14B-Instruct-1M",
serving_model = None
def get_model(model="deepseek-ai/deepseek-llm-7b-chat"):
    global serving_model
    if serving_model is not None:
        return serving_model
    # if model too large, increase tp/pp size
    llm = LLM(model=model, tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.5, max_tokens=8192)
    def model_call(messages, instanced_model, sampling_params):
        outputs = instanced_model.chat(messages,
                   sampling_params=sampling_params,
                   use_tqdm=False)
        #chat_template_content_format='openai',
        return outputs
    serving_model = partial(model_call, instanced_model=llm, sampling_params=sampling_params)
    return serving_model
    return partial(model_call, instanced_model=llm, sampling_params=sampling_params)
 
def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)
 
if __name__ == "__main__":
    print("=" * 80)
 
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content": "Tell me the rule of the settlers of catan and what strategy do you think is the best to win the game..",
        },
    ]
 
    # multiple conversations
    conversations = [conversation for _ in range(10)]
 
    model = get_model()
    outputs = model(conversations)
 
    print(outputs)
    print_outputs(outputs)
 
    # A chat template can be optionally supplied.
    # If not, the model will use its default chat template.
 
    # with open('template_falcon_180b.jinja', "r") as f:
    #     chat_template = f.read()
 
    # outputs = llm.chat(
    #     conversations,
    #     sampling_params=sampling_params,
    #     use_tqdm=False,
    #     chat_template=chat_template,
    # )