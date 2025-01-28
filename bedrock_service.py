import boto3
import json
import logging
from typing import Dict, Optional, List
from botocore.config import Config
import string

class BedrockService:
    def __init__(self, model_config: Dict):
        """Initialize Bedrock service with model configuration."""
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize Bedrock client with retry configuration."""
        config = Config(
            retries=dict(
                max_attempts=3,
                mode='standard'
            )
        )

        return boto3.client(
            service_name='bedrock-runtime',
            region_name=self.model_config.get('region', 'us-east-1'),
            config=config
        )
    
    def invoke_nova_model(self, prompt: str) -> Optional[str]:
        """Invoke Amazon Nova Pro model for analysis."""
        try:
            # Convert prompt to single line to avoid formatting issues
            prompt = " ".join(prompt.split())

            # Format request body based on Nova Pro specifications
            body = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            })

            response = self.client.invoke_model(
                modelId=self.model_config.get('nova_model_id', 'amazon.nova-pro-v1:0'),
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            # Parse response
            response_body = json.loads(response.get('body').read())
            print(f"Response Body:{response_body}")
            
            # Extract content based on Nova Pro response format
            if response_body and 'output' in response_body:
                message = response_body['output'].get('message', {})
                if 'content' in message and len(message['content']) > 0:
                    return message['content'][0].get('text', '')
            
            self.logger.error(f"Unexpected Nova Pro response format: {response_body}")
            return None

        except Exception as e:
            self.logger.error(f"Error invoking Nova Pro model: {str(e)}")
            print(f"Full error details: {str(e)}")  # Debug print
            return None
    
    def invoke_model_simple(self, prompt: str, system_prompt: str = None) -> Optional[Dict]:
        """Invoke Claude with a simple prompt."""
        try:
            # Format messages in the correct structure
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Add system prompt if provided
            if system_prompt:
                messages.insert(0, {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                })

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": self.model_config['max_tokens'],
                "temperature": self.model_config['temperature'],
                "top_p": self.model_config['top_p']
            })
            
            modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
            accept = 'application/json'
            contentType = 'application/json'
            
            response = self.client.invoke_model(
                modelId=modelId,
                body=body,
                contentType=contentType,
                accept=accept
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body["content"][0]["text"]
            
        except Exception as e:
            self.logger.error(f"Error invoking Claude simple: {str(e)}")
            return None

    def invoke_model_summary_analysis(self, prompt: str, system_prompt: str = None) -> Optional[Dict]:
        """Invoke Claude 3.5 Sonnet with the given prompt."""
        try:
            print("Inside bedrock summary analysis")
            messages = []
            
            # Add user prompt first (system prompt is handled as a user message)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            })

            # Prepare request body
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.model_config['max_tokens'],
                "messages": messages,
                "temperature": self.model_config['temperature'],
                "top_p": self.model_config['top_p']
            })

            #print("Request body:", body)  # Debug print

            # Make API call
            response = self.client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read().decode())
            #print("Raw response:", response_body)  # Debug print

            # Verify response structure
            if 'content' not in response_body or not response_body['content']:
                print("Invalid response structure:", response_body)
                return None

            try:
                # Try to parse the content as JSON if it's meant to be JSON
                content_text = response_body['content'][0]['text']
                content_json = json.loads(content_text)
                return {
                    'content': content_text,
                    'data': content_json.get('data', []),
                    'model': response_body.get('model'),
                    'usage': response_body.get('usage', {})
                }
            except json.JSONDecodeError:
                # If not JSON, return as regular response
                return {
                    'content': response_body['content'][0]['text'],
                    'model': response_body.get('model'),
                    'usage': response_body.get('usage', {})
                }

        except Exception as e:
            self.logger.error(f"Error invoking Claude summary: {str(e)}")
            print(f"Full error details: {str(e)}")  # Debug print
            return None
        
    def invoke_model(self, messages: List[Dict]) -> Optional[Dict]:
        """Invoke Claude with the given messages."""
        try:
            # Convert system and user messages to proper format
            formatted_messages = []
            for message in messages:
                if message["role"] == "system":
                    # For system messages, add to instructions parameter instead
                    formatted_messages.append({
                        "role": "user",
                        "content": message["content"]
                    })
                else:
                    formatted_messages.append({
                        "role": "user" if message["role"] == "user" else "assistant",
                        "content": message["content"]
                    })

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.model_config['max_tokens'],
                "messages": formatted_messages,
                "temperature": self.model_config['temperature'],
                "top_p": self.model_config['top_p']
            })

            response = self.client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Updated model ID
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read().decode())
            
            return {
                'content': response_body['content'][0]['text'],
                'model': response_body.get('model'),
                'usage': response_body.get('usage', {})
            }

        except Exception as e:
            self.logger.error(f"Error invoking Claude: {str(e)}")
            return None

    def _format_chunks_for_context(self, chunks: List[Dict]) -> str:
        """Format document chunks for context."""
        context_parts = []
        for chunk in chunks:
            source = os.path.basename(chunk['source'])
            # Extract section information from the content
            content = chunk['content']
            section = "Unknown Section"  # Default value
            
            # Try to identify section from content structure
            section_patterns = [
                r"Section\s*\d+[\.:]\s*([^\n]+)",  # Matches "Section X: Title"
                r"^\s*(\d+\.\s*[^\n]+)",           # Matches numbered sections
                r"^([A-Z][^\n]+?)\s*\n",           # Matches capitalized headers
            ]
            
            for pattern in section_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    section = matches[0].strip()
                    break

            # Extract page number if available
            page = chunk.get('page_number', 'N/A')
            if isinstance(page, str) and page.lower() == 'n/a':
                # Try to find page number in content or metadata
                page_matches = re.findall(r'Page\s*(\d+)', content, re.IGNORECASE)
                if page_matches:
                    page = page_matches[0]
                else:
                    # Check metadata
                    page = chunk.get('metadata', {}).get('page_number', 'N/A')

            context_parts.append(
                f"""Document: {source}
                Page: {page}
                Section: {section}
                Content: {content}
                ---"""
            )
        return "\n\n".join(context_parts)

    def format_prompt_for_document_qa(self, context: str, question: str) -> List[Dict]:
        """Format prompt specifically for document QA with Claude."""
        instructions = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """You are a senior financial analyst specializing in banking and financial institutions. Your role is to:

    1. Analyze and respond as a financial expert, providing deep insights into financial results and metrics.
    2. Base your analysis strictly on the provided document context - do not make assumptions or include external information.
    3. Structure your response professionally with these sections:
    - Executive Summary: Brief overview of key findings along with the reasoning
    - Detailed Analysis: In-depth examination of the metrics
    - Key Observations: Important trends or notable points
    - Sources: List of all citations used

    4. For every piece of financial information:
    - Include specific page numbers AND sections in citations [Document Name (Page X, Section Y)]
    - Explain the significance of the numbers
    - Compare periods when relevant
    - Note any important trends or changes

    5. IMPORTANT: For every citation:
    - Always include both the page number AND section
    - If page number is truly N/A, explain why
    - Use format: [Document Name (Page X, Section Y)]
    - List complete citation details in Sources section

    6. If the information requested is not in the document:
    - Clearly state that the information is not available in the provided document
    - Explain what related information is available, if any
    - Suggest what other documents might be helpful

    Remember to:
    - Maintain a professional, analytical tone
    - Cite every factual statement with page and section
    - Explain financial terms and metrics when relevant
    - Focus on meaningful analysis rather than just stating numbers"""
                }
            ]
        }
        
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Context:
    {context}

    Question: {question}

    Please provide a comprehensive financial analysis with proper citations including both page numbers and sections."""
                }
            ]
        }

        return [instructions, user_message]

    def process_document_qa_response(self, response: Dict) -> str:
        """Process Claude's response for document QA."""
        if not response or 'content' not in response:
            return "I apologize, but I encountered an error generating the response."

        content = response['content']

        # Format usage statistics
        if 'usage' in response:
            usage = response['usage']
            content += f"\n\n*Response generated using {usage.get('input_tokens', 0)} input tokens and {usage.get('output_tokens', 0)} output tokens.*"

        return content

    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            test_prompt = "Test connection"
            response = self.invoke_model(test_prompt)
            return response is not None
        except Exception:
            return False