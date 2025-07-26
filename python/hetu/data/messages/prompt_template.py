from hetu.data.messages.utils import compile_jinja_template, render_template

# Examples of Jinja templates
CHATML_TEMPLATE = """
{%- for message in messages -%}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor -%}
"""

SUMMARIZE_TEMPLATE = """
{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{{ "Summarize this dialogue:\n" + message['content'] + "\n---\nSummary:\n" }}
{%- else -%}
{{ message['content'] }}
{%- endif -%}
{%- endfor -%}
"""

QWESTION_ANSWER_TEMPLATE = """
{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{{ "Question: " + message['content'] + "\n\nAnswer: " }}
{%- else -%}
{{ message['content'] }}
{%- endif -%}
{%- endfor -%}
"""

class PromptTemplate(object):
    template: str
    
    def __init__(
        self,
        template: str,
        **kwargs,
    ):
        """
        Initialize a PromptTemplate with a Jinja2 template string.
        
        Args:
            template (str): The Jinja2 template string.
            **kwargs: Additional keyword arguments.
        """
        self.template = template
    
    def __call__(
        self,
        messages,
        return_mask: bool = True,
        train_on_all_assistant: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        """
        Format a list of messages according to the template.
        
        Args:
            messages (List[Dict]): List of message dictionaries, each with 'role' and 'content' keys.
            return_mask (bool, optional): Whether to return position masks for training. Defaults to True.
            train_on_all_assistant (bool, optional): Whether to train on all assistant messages. Defaults to True.
            add_generation_prompt (bool, optional): Whether to add a generation prompt at the end. Defaults to False.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Union[str, Tuple[str, List[Tuple[int, int]]]]: 
                - If return_mask=False: the formatted message string.
                - If return_mask=True: a tuple of (formatted string, list of content position tuples).
                  Each position tuple contains (start_position, end_position) for tracked message content,
                  used for building label mask.
        """
        compiled_template = compile_jinja_template(self.template, tracking=return_mask)
        context = {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt
        }
        return render_template(compiled_template, context, tracking=return_mask, train_on_all_assistant=train_on_all_assistant)
