import logging
from jinja2 import Environment
from functools import lru_cache

@lru_cache
def compile_jinja_template(template_str, tracking=True):
    """
    Compile a Jinja2 template string and cache the result.
    
    Args:
        template_str (str): The template string to compile.
        tracking (bool, optional): Whether to enable tracking for the template. Defaults to True.
        
    Returns:
        jinja2.Template: The compiled Jinja2 template.
    """
    logging.debug(f"Compiling template with tracking={tracking}")
    env = Environment(keep_trailing_newline=True)
    template = env.from_string(template_str)
    logging.debug("Template compiled successfully")
    return template

def render_template(template, context, tracking=True, train_on_all_assistant=True):
    """
    Render a Jinja2 template with the given context, optionally tracking message content positions.
    
    Args:
        template (jinja2.Template): The compiled Jinja2 template to render.
        context (dict): The context dictionary for template rendering.
        tracking (bool, optional): Whether to track message content positions. Defaults to True.
        train_on_all_assistant (bool, optional): Whether to train on all assistant messages. Defaults to True.
        
    Returns:
        Union[str, Tuple[str, List[Tuple[int, int]]]]: 
            - If tracking=False: the rendered template string.
            - If tracking=True: a tuple of (rendered string, list of content position tuples).
              Each position tuple contains (start_position, end_position) for tracked message content.
    """
    logging.debug(f"Rendering template with tracking={tracking}, train_on_all_assistant={train_on_all_assistant}")
    if not tracking:
        return template.render(**context)
    
    tracking_data = {
        'message_blocks': [],
        'current_position': 0,
        'train_on_all_assistant': train_on_all_assistant,
        'messages': context.get('messages', [])
    }
    
    class OutputTracker:
        """
        A helper class that tracks positions of text during template rendering.
        
        This class is used internally by the render_template function to track the positions
        of different message contents within the rendered output.
        """
        
        def __init__(self):
            """
            Initialize an OutputTracker instance.
            """
            self.buffer = []
            self.position = 0
        
        def append(self, text: str):
            """
            Append text to the buffer and track its position.
            
            Args:
                text (str): The text to append.
            """
            start_pos = self.position
            self.buffer.append(text)
            self.position += len(text)
            tracking_data['current_position'] = self.position
                
            self._check_message_content(text, start_pos, self.position)
        
        def _check_message_content(self, text, start, end):
            """
            Check if the text contains any message content that should be tracked.
            
            Args:
                text (str): The text to check.
                start (int): The start position of the text.
                end (int): The end position of the text.
            """
            for msg in tracking_data['messages']:
                if (
                    msg.get('masked') == False or
                    (tracking_data['train_on_all_assistant'] and msg.get('role') == 'assistant')
                ):
                    content = msg.get('content', '')
                    if text == content or content in text:
                        if content in text:
                            content_start = text.find(content)
                            content_end = content_start + len(content)
                            abs_start = start + content_start
                            abs_end = start + content_end
                        else:
                            abs_start = start
                            abs_end = end
                        
                        tracking_data['message_blocks'].append({
                            "message_id": id(msg),
                            "role": msg.get("role"),
                            "start": abs_start,
                            "end": abs_end
                        })
                        logging.debug(f"Tracked message content: role={msg.get('role')}, position={abs_start}-{abs_end}")
        
        def get_value(self):
            """
            Get the entire rendered content as a string.
            
            Returns:
                str: The complete rendered content.
            """
            return "".join(self.buffer)
    
    tracker = OutputTracker()
    logging.debug("Output tracker initialized")
    
    env = template.environment
    original_concat = env.concat
    def tracked_concat(items):
        result = []
        for item in items:
            if item is None:
                continue
            tracker.append(item)
            result.append(item)
        return "".join(result)
    
    try:
        logging.debug("Replacing environment concat function with tracking version")
        env.concat = tracked_concat
        template.render(**context)
        logging.debug("Template rendering completed")
        rendered_content = tracker.get_value()
        logging.debug(f"Rendered content length: {len(rendered_content)}")
    finally:
        env.concat = original_concat
        logging.debug("Restored original concat function")
    
    logging.debug(f"Messages in context: {[m.get('role') for m in context.get('messages', [])]}")
    logging.debug(f"Tracked blocks count: {len(tracking_data['message_blocks'])}")
    for idx, block in enumerate(tracking_data['message_blocks']):
        logging.debug(f"Block {idx}: role={block.get('role')}, start={block.get('start')}, end={block.get('end')}")

    blocks = []
    for block in tracking_data['message_blocks']:
        if block["end"] is not None:
            blocks.append((block["start"], block["end"]))
            logging.debug(f"Extracted block: ({block['start']}, {block['end']})")
    
    merged_blocks = []
    if blocks:
        sorted_blocks = sorted(blocks, key=lambda x: x[0])
        logging.debug(f"Sorted blocks: {sorted_blocks}")
        
        # merge overlapping blocks
        merged = [sorted_blocks[0]]
        for current in sorted_blocks[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
                logging.debug(f"Merged overlapping blocks: {last} + {current} -> {merged[-1]}")
            else:
                merged.append(current)
                logging.debug(f"Added non-overlapping block: {current}")
        merged_blocks = merged
    
    logging.debug(f"Final merged blocks: {merged_blocks}")
    return rendered_content, merged_blocks