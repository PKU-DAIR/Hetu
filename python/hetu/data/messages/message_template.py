from typing import Any, Mapping, Protocol, Dict, Optional, List

class MessageTemplate(Protocol):
    def __call__(self, sample: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        """
        Protocol for message templates that convert a dataset sample to a list of messages.
        
        Args:
            sample (Mapping[str, Any]): A sample from the dataset.
            
        Returns:
            List[Mapping[str, Any]]: A list of message dictionaries, each with 'role', 'content', and 'masked' keys.
        """
        pass

class InputOutputTemplate(MessageTemplate):
    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        """
        Initialize an InputOutputTemplate that formats a dataset sample as a user-assistant conversation.
        
        Args:
            train_on_input (bool, optional): Whether to train on the input messages. Defaults to False.
            column_map (Optional[Dict[str, str]], optional): Mapping from standard column names to dataset column names.
                                                           Defaults to None, which means {'input': 'input', 'output': 'output'}.
            new_system_prompt (Optional[str], optional): System prompt to add at the beginning. Defaults to None.
            
        Raises:
            ValueError: If the column_map doesn't contain 'input' or 'output' keys.
        """
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt

        self.column_map = column_map

        if self.column_map is not None:
            if "input" not in self.column_map:
                raise ValueError(
                    f"Expected a key of 'input' in column_map but found {self.column_map.keys()}."
                )
            if "output" not in self.column_map:
                raise ValueError(
                    f"Expected a key of 'output' in column_map but found {self.column_map.keys()}."
                )
        else:
            self.column_map = {"input": "input", "output": "output"}
    
    def __call__(self, sample: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        """
        Convert a dataset sample to a list of messages in a user-assistant conversation format.
        
        Args:
            sample (Mapping[str, Any]): A sample from the dataset with 'input' and 'output' keys
                                      (or the keys specified in column_map).
            
        Returns:
            List[Mapping[str, Any]]: A list of message dictionaries, each with 'role', 'content', and 'masked' keys.
                                    The list contains a user message with the input and an assistant message with the output,
                                    and optionally a system message if new_system_prompt is provided.
        """
        content = sample[self.column_map["input"]]
        output_content = sample[self.column_map["output"]]    
        
        messages = [
            {
                "role": "user",
                "content": content,
                "masked": not self.train_on_input,
            },
            {
                "role": "assistant",
                "content": output_content,
                "masked": False,
            },
        ]
        
        if self.new_system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": self.new_system_prompt,
                    "masked": True,
                },
                *messages,
            ]
        
        return messages

class AlpacaTemplate(MessageTemplate):
    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize an AlpacaTemplate that formats a dataset sample in the Alpaca instruction format.
        
        Args:
            train_on_input (bool, optional): Whether to train on the input messages. Defaults to False.
            column_map (Optional[Dict[str, str]], optional): Mapping from standard column names to dataset column names.
                                                           Defaults to None.
        """
        self.train_on_input = train_on_input
        self.column_map = column_map
        self.template = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }
    
    def __call__(self, sample: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        """
        Convert a dataset sample to a list of messages in the Alpaca instruction format.
        
        Args:
            sample (Mapping[str, Any]): A sample from the dataset with 'instruction', 'input' (optional),
                                      and 'output' keys (or the keys specified in column_map).
            
        Returns:
            List[Mapping[str, Any]]: A list of message dictionaries, each with 'role', 'content', and 'masked' keys.
                                    The list contains a user message with the formatted instruction+input
                                    and an assistant message with the output.
        """
        column_map = self.column_map or {}
        key_input = column_map.get("input", "input")
        key_instruction = column_map.get("instruction", "instruction")
        key_output = column_map.get("output", "output")

        if key_input in sample and sample[key_input]:
            prompt = self.template["prompt_input"].format(
                instruction=sample[key_instruction], input=sample[key_input]
            )
        else:
            prompt = self.template["prompt_no_input"].format(
                instruction=sample[key_instruction]
            )
        
        messages = [
            {
                "role": "user",
                "content": prompt,
                "masked": not self.train_on_input,
            },
            {
                "role": "assistant",
                "content": sample[key_output],
                "masked": False,
            },
        ]
        
        return messages

class ShareGPTTemplate(MessageTemplate):
    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        """
        Initialize a ShareGPTTemplate that formats a dataset sample in the ShareGPT conversation format.
        
        Args:
            train_on_input (bool, optional): Whether to train on the input messages. Defaults to False.
            column_map (Optional[Dict[str, str]], optional): Mapping from standard column names to dataset column names.
                                                           Defaults to None, which means {'conversations': 'conversations'}.
            new_system_prompt (Optional[str], optional): System prompt to add at the beginning. Defaults to None.
            
        Raises:
            ValueError: If the column_map doesn't contain a 'conversations' key.
        """
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "conversations" not in column_map:
                raise ValueError(
                    f"Expected a key of 'conversations' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"conversations": "conversations"}

    def __call__(self, sample: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        """
        Convert a dataset sample to a list of messages in the ShareGPT format.
        
        Args:
            sample (Mapping[str, Any]): A sample from the dataset with a 'conversations' key
                                      (or the key specified in column_map) containing a list of
                                      message objects with 'from' and 'value' keys.
            
        Returns:
            List[Mapping[str, Any]]: A list of message dictionaries, each with 'role', 'content', and 'masked' keys.
                                    Maps 'from' values ('system', 'human', 'gpt') to corresponding roles.
        """
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        messages = []
        if self.new_system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": self.new_system_prompt,
                    "masked": True,
                }
            )

        for message in sample[self._column_map["conversations"]]:
            role = role_map[message["from"]]
            content = message["value"]
            masked = not (role == "assistant")
            if role == "system" and self.new_system_prompt is not None:
                continue

            messages.append(
                {
                    "role": role,
                    "content": content,
                    "masked": masked,
                }
            )

        return messages

class OpenAITemplate(MessageTemplate):
    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        """
        Initialize an OpenAITemplate that formats a dataset sample in the OpenAI API conversation format.
        
        Args:
            train_on_input (bool, optional): Whether to train on the input messages. Defaults to False.
            column_map (Optional[Dict[str, str]], optional): Mapping from standard column names to dataset column names.
                                                           Defaults to None, which means {'messages': 'messages'}.
            new_system_prompt (Optional[str], optional): System prompt to add at the beginning. Defaults to None.
            
        Raises:
            ValueError: If the column_map doesn't contain a 'messages' key.
        """
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "messages" not in column_map:
                raise ValueError(
                    f"Expected a key of 'messages' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"messages": "messages"}

    def __call__(self, sample: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        """
        Convert a dataset sample to a list of messages in the OpenAI API format.
        
        Args:
            sample (Mapping[str, Any]): A sample from the dataset with a 'messages' key
                                      (or the key specified in column_map) containing a list of
                                      message objects with 'role' and 'content' keys.
            
        Returns:
            List[Mapping[str, Any]]: A list of message dictionaries, each with 'role', 'content', and 'masked' keys.
                                    The content is extracted from either a string or a nested object with a 'text' key.
        """
        updated_messages = []
        if self.new_system_prompt is not None:
            updated_messages.append(
                {
                    "role": "system",
                    "content": self.new_system_prompt,
                    "masked": True,
                }
            )
        for message in sample[self._column_map["messages"]]:
            role = message["role"]
            if role == "system" and self.new_system_prompt is not None:
                continue
            trained = (role == "assistant") or self.train_on_input
            if isinstance(message["content"], list):
                content = message["content"]["text"]
            elif isinstance(message["content"], str):
                content = message["content"]
            updated_messages.append(
                {
                    "role": role,
                    "content": content,
                    "masked": not trained,
                }
            )

        return updated_messages