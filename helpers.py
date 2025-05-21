import os
import logging


logger = logging.getLogger(__name__)
os.environ["HUGGINGFACE_API_KEY"] = "please_input_your_api_key"

def get_llm(llm_type, model_name):
    
    if llm_type == "openai":
        from langchain_community.chat_models import ChatOpenAI
        from langchain_core.messages import HumanMessage
        import tiktoken

        os.environ["OPENAI_API_KEY"] = "please_input_your_api_key"

        if model_name == "gpt-3.5-turbo":
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            MAX_TOKENS = 16000

            class SafeChatOpenAI(ChatOpenAI):
                def invoke(self, messages):
                    try:
                        user_msg = messages[-1].content if messages else ""
                        token_len = len(encoding.encode(user_msg))
                        print(f"[Row] Token length: {token_len}")
                        if token_len > MAX_TOKENS:
                            logger.warning(f"Skipping: {token_len} tokens")
                            return type("Response", (), {"content": "[Skipped: Too Long]"})()
                        return super().invoke(messages)
                    except Exception as e:
                        logger.error(f"Invoke error: {e}")
                        return type("Response", (), {"content": "[Error] " + str(e)})()

            return SafeChatOpenAI(model=model_name, temperature=0.3)

        else:
            return ChatOpenAI(model=model_name, temperature=0.3)


    
    elif llm_type == "llama":
      
        def llama_invoke(messages, model_name=model_name, temperature=0.3):
            prompt = messages[0].content if messages else ""

            api_url = f"https://api-inference.huggingface.co/{model_name}"

            headers = {
                "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
                "Content-Type": "application/json",
            }

            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "return_full_text": False,
                },
                "options": {
                    "wait_for_model": True
                }
            }

            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                generated_text = response.json()[0]["generated_text"]
                return type("Response", (), {"content": generated_text})()
            except Exception as e:
                logger.error(f"HuggingFace Llama API failed: {e}")
                raise

        return type("LLMWrapper", (), {
            "invoke": staticmethod(llama_invoke)
        })()
    elif llm_type == "deepseek":
     
        from transformers import pipeline

        def deepseek_invoke(messages, model_name=model_name, temperature=0.7):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            prompt = messages[0].content if messages else ""

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                return_full_text=False,
                max_new_tokens=256,
            )

            try:
                response = text_generator(prompt)
                return type("Response", (), {"content": response[0]["generated_text"]})()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Transformers DeepSeek failed: {e}")
                return type("Response", (), {"content": "[Error] " + str(e)})()

        return type("LLMWrapper", (), {
            "invoke": staticmethod(deepseek_invoke)
        })()
