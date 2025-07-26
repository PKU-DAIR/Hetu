
class HetuMLLMProcessor():
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def __call__(self, images, texts, videos):
        if images is not None:
            image_inputs = self.image_processor.preprocess(images=images, videos=None)
            image_grid_thw = image_inputs["image_grid_thws"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor.preprocess(images=None, videos=videos)
            video_grid_thw = videos_inputs["video_grid_thws"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(texts, list):
            texts = [texts]

        if image_grid_thw is not None:
            index = 0
            for i in range(len(texts)):
                while self.image_token in texts[i]:
                    texts[i] = texts[i].replace(
                        self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod()), 1
                    )
                    index += 1
                texts[i] = texts[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(texts)):
                while self.video_token in texts[i]:
                    texts[i] = texts[i].replace(
                        self.video_token, "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                texts[i] = texts[i].replace("<|placeholder|>", self.video_token)

        text_inputs = []
        for text in texts:
            text_inputs.append(self.tokenizer.tokenize(text))

        return text_inputs, image_inputs, videos_inputs