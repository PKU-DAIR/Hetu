from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional


class InstructTemplate(ABC):
    """
    Interface for instruction templates. Each template should include the template
    prompt with placeholders for the data inputs.
    """

    template = ""

    @classmethod
    @abstractmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Format the prompt template with the given arguments.

        Args:
            sample (Mapping[str, Any]): a single data sample with various fields
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical. Note: if the sample output is not named
                as "output" in the dataset, you always need to map it to "output" in column_map.

        Returns:
            The formatted prompt
        """
        pass

class StackExchangedPairedwithContextTemplate(InstructTemplate):
    """
    Prompt template for preference datasets similar to StackExchangedPaired.

    .. code-block:: text
    
        Label: <YOUR LABEL HERE>

        Question: <YOUR QUESTION HERE>

        Answer:
    """
    
    template = "{context}QUESTION: {question}\n ### ANSWER (yes|no|maybe): "
    
    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        column_map = column_map or {}
        def get_context(query):
            context = ""
            for i, label in enumerate(query['context']['labels']):
                context += f"{label}: {query['context']['contexts'][i]}\n"
            return context
        context = get_context(sample)
        column_map = column_map or {}
        key_question = column_map.get("question", "question")
        prompt = cls.template.format(context=context, question=sample[key_question])
        return prompt

class AlpacaInstructTemplate(InstructTemplate):
    """
    Prompt template for Alpaca-style datasets. Template prompt changes slightly depending
    on if there's an instruction + input or just an instruction.

    .. code-block:: text

        Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.

        ### Instruction:
        <YOUR INSTRUCTION HERE>

        ### Input:
        <YOUR INPUT HERE>

        ### Response:


    Or without 'input'

    .. code-block:: text

        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        <YOUR INSTRUCITON HERE>

        ### Response:


    """

    template = {
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

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample with instruction
            column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names
                in the template to the column names in the sample. If None, assume these are identical.

        Examples:
            >>> # Simple instruction
            >>> AlpacaInstructTemplate.format(sample={"instruction": "Write a poem"})
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\\n\\n### Instruction:\\nWrite a poem\\n\\n### Response:\\n

            >>> # Instruction with input
            >>> AlpacaInstructTemplate.format(sample={"instruction": "Write a poem", "input": "The poem should be 5 lines long"})
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\\n\\n### Instruction:\\nWrite a poem\\n\\n### Input:\\n
            The poem should be 5 lines long\\n\\n### Response:\\n

            >>> # Instruction with column map where the 'instruction' key is actually named 'prompt' in the given sample
            >>> AlpacaInstructTemplate.format(sample={"prompt": "Write me a poem"}, column_map={"instruction": "prompt"})
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\\n\\n### Instruction:\\nWrite a poem\\n\\n### Response:\\n

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_input = column_map.get("input", "input")
        key_instruction = column_map.get("instruction", "instruction")
        def get_input(query):
            if query.find('\n') == -1:
                return ''
            return '\n'.join(query.split('\n')[1:])

        if key_input in sample and sample[key_input]:
            prompt = cls.template["prompt_input"].format(
                instruction=sample[key_instruction], input=sample[key_input]
            )
            # prompt = cls.template["prompt_input"].format(
            #     instruction=sample['query'].split('\n')[0], input=get_input(sample['query'])
            # )
        else:
            prompt = cls.template["prompt_no_input"].format(
                instruction=sample[key_instruction]
            )
            # prompt = cls.template["prompt_no_input"].format(
            #     instruction=sample['query'].split('\n')[0]
            # )
        return prompt

class SummarizeTemplate(InstructTemplate):
    """
    Prompt template to format datasets for summarization tasks.

    .. code-block:: text

        Summarize this dialogue:
        <YOUR DIALOGUE HERE>
        ---
        Summary:

    """

    template = "Summarize this dialogue:\n{dialogue}\n---\nSummary:\n"

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from dialogue.

        Args:
            sample (Mapping[str, Any]): a single data sample with dialog
            column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names
                in the template to the column names in the sample. If None, assume these are identical.

        Examples:
            >>> # Simple dialogue
            >>> SummarizeTemplate.format(sample={"dialogue": "Hello, how are you? Did you know the capital of France is Paris?"})
            Summarize this dialogue:
            Hello, how are you? Did you know the capital of France is Paris?
            ---
            Summary:

            >>> # Dialogue with column map where the 'dialogue' key is actually named 'prompt' in the given sample
            >>> SummarizeTemplate.format(
            ...     sample={"prompt": "Hello, how are you? Did you know the capital of France is Paris?"},
            ...     column_map={"dialogue": "prompt"}
            ... )
            Summarize this dialogue:
            Hello, how are you? Did you know the capital of France is Paris?
            ---
            Summary:

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_dialogue = column_map.get("dialogue", "dialogue")

        prompt = cls.template.format(dialogue=sample[key_dialogue])
        return prompt


class StackExchangedPairedTemplate(InstructTemplate):
    """
    Prompt template for preference datasets similar to StackExchangedPaired.

    .. code-block:: text

        Question: <YOUR QUESTION HERE>

        Answer:
    """

    template = "Question: {question}\n\nAnswer: "

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample with instruction
            column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names
                in the template to the column names in the sample. If None, assume these are identical.

        Examples:
            >>> # Simple question
            >>> StackExchangedPairedTemplate.format(sample={"question": "What is the capital of France?"})
            Question: What is the capital of France?\\n\\nAnswer:

            >>> # Question with column map where the 'question' key is actually named 'prompt' in the given sample
            >>> StackExchangedPairedTemplate.format(
            ...     sample={"prompt": "What is the capital of France?"},
            ...     column_map={"question": "prompt"}
            ... )
            Question: What is the capital of France?\\n\\nAnswer:

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_prompt = column_map.get("question", "question")
        prompt = cls.template.format(question=sample[key_prompt])

        return prompt