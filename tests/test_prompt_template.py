from hetu.data.messages.prompt_template import PromptTemplate
from hetu.data.messages.prompt_template import CHATML_TEMPLATE

def test_prompt_template():
    chatml_template = PromptTemplate(CHATML_TEMPLATE)
    messages = [
        {
            "role": "user",
            "content": "Hello, how are you?",
            "masked": True,
        },
        {
            "role": "assistant",
            "content": "I'm fine, thank you.",
            "masked": False,
        },
    ]

    print(chatml_template(messages))

if "__main__" == __name__:
    test_prompt_template()