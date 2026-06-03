from urllib import response

from huggingface_hub import InferenceClient

from config import HF_TOKEN

MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LLMClient:

    def __init__(self):

        from token_manager import TokenManager

        self.token_manager = TokenManager()

        self.client = InferenceClient(
            api_key=self.token_manager.get_token()
        )
    def _rebuild_client(self):

        self.client = InferenceClient(
            api_key=self.token_manager.get_token()
        )
    # def generate(
    #     self,
    #     system_prompt,
    #     user_prompt
    # ):

    #     response = self.client.chat.completions.create(
    #         model=MODEL,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": system_prompt
    #             },
    #             {
    #                 "role": "user",
    #                 "content": user_prompt
    #             }
    #         ],
    #         temperature=0.1,
    #         max_tokens=1500
    #     )

    #     return response.choices[0].message.content
    
    def generate(
        self,
        system_prompt,
        user_prompt
    ):

        attempts = len(
            self.token_manager.tokens
        )

        last_error = None

        for _ in range(attempts):

            try:
                print("SENDING TO HF")
                response = (
                    self.client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        temperature=0.1,
                        max_tokens=1500
                    )
                )
                print("HF RETURNED")
                print(type(response))

                print("EXTRACTING CONTENT")

                print(response)
                return (
                    response
                    .choices[0]
                    .message.content
                )

            except Exception as e:

                msg = str(e)

                if (
                    "402" in msg
                    or "Payment Required" in msg
                    or "quota" in msg.lower()
                ):

                    print(
                        "\nToken exhausted."
                    )

                    self.token_manager.rotate()

                    self._rebuild_client()

                    continue

                last_error = e
                break

        raise last_error