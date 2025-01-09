import json
from typing import Dict, List, Optional, Tuple
import boto3
from datetime import datetime
import re

class GuardrailsService:
    def __init__(self, config: Dict):
        """Initialize Guardrails Service with configuration."""
        self.config = config
        self.content_filters = self._setup_content_filters()
        self.max_tokens = config['model_config']['max_tokens']
        self.allowed_domains = config['security_config']['allowed_domains']
        
        # Initialize AWS Comprehend client
        self.comprehend = boto3.client('comprehend',
            region_name=config['aws_config']['region'])
        
        # PII entity types to detect
        self.pii_entity_types = config['security_config']['guardrails']['pii_detection']['entity_types']
        self.redaction_enabled = config['security_config']['guardrails']['pii_detection']['redaction_enabled']

    def _setup_content_filters(self) -> Dict[str, List[str]]:
        """Setup content filtering rules."""
        return {
            'blocked_keywords': self.config['security_config']['guardrails']['content_filtering']['blocked_keywords'],
            'sensitive_commands': self.config['security_config']['guardrails']['content_filtering']['sensitive_commands']
        }

    def detect_pii(self, text: str) -> Tuple[List[Dict], str]:
        """
        Detect PII entities using AWS Comprehend and regex patterns.
        Returns tuple of (detected_entities, redacted_text)
        """
        try:
            detected_entities = []
            redacted_text = text
            
            # Regular expression for credit card numbers
            credit_card_pattern = r'\b(?:\d{4}[- ]){3}\d{4}\b|\b\d{16}\b'
            
            # Check for credit card numbers first
            cc_matches = list(re.finditer(credit_card_pattern, text))
            for match in reversed(cc_matches):
                detected_entities.append({
                    'Type': 'CREDIT_CARD_NUMBER',
                    'Score': 1.0,
                    'BeginOffset': match.start(),
                    'EndOffset': match.end()
                })
                if self.redaction_enabled:
                    redacted_text = (
                        redacted_text[:match.start()] +
                        "[REDACTED-CREDIT_CARD_NUMBER]" +
                        redacted_text[match.end():]
                    )
            
                response = self.comprehend.detect_pii_entities(
                    Text=text,
                    LanguageCode='en'
                )
            
            # Sort entities by offset in reverse order to handle redaction
            entities = sorted(response['Entities'], key=lambda x: x['BeginOffset'], reverse=True)
            
            for entity in entities:
                entity_type = entity['Type']
                if entity_type in self.pii_entity_types:
                    detected_entities.append({
                        'Type': entity_type,
                        'Score': entity['Score'],
                        'BeginOffset': entity['BeginOffset'],
                        'EndOffset': entity['EndOffset']
                    })
                    
                    # Perform redaction if enabled
                    if self.redaction_enabled:
                        redacted_text = (
                            redacted_text[:entity['BeginOffset']] +
                            f"[REDACTED-{entity_type}]" +
                            redacted_text[entity['EndOffset']:]
                        )
            
            return detected_entities, redacted_text

        except Exception as e:
            self.log_security_event(
                "pii_detection_error",
                {"error": str(e)}
            )
            return [], text

    def validate_prompt(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user prompt against security rules.
        Returns (is_valid, error_message).
        """
        try:
            # Check for empty or too long prompts
            if not prompt or not prompt.strip():
                return False, "Empty prompt", None
            
            if len(prompt) > self.max_tokens:
                return False, f"Prompt exceeds maximum length of {self.max_tokens} tokens", None

            # Check for PII using AWS Comprehend
            detected_entities, redacted_prompt = self.detect_pii(prompt)
            
            if detected_entities:
                entity_types = [entity['Type'] for entity in detected_entities]
                # Always block credit card numbers
                if 'CREDIT_CARD_NUMBER' in entity_types:
                    return False, "Credit card information is not allowed for security reasons", None
                elif self.redaction_enabled:
                    # Allow other PII but with redaction
                    return True, f"PII detected and redacted: {', '.join(set(entity_types))}", redacted_prompt
                else:
                    # Block the prompt if redaction is disabled
                    return False, f"Found potential PII: {', '.join(set(entity_types))}", None

            # Check for blocked content
            blocked_content = self._check_blocked_content(prompt)
            if blocked_content:
                return False, f"Found blocked content: {', '.join(blocked_content)}", None

            # Check for SQL injection attempts
            if self._check_sql_injection(prompt):
                return False, "Potential SQL injection detected", None

            # Check for command injection attempts
            if self._check_command_injection(prompt):
                return False, "Potential command injection detected", None

            return True, None, prompt

        except Exception as e:
            self.log_security_event(
                "prompt_validation_error",
                {"error": str(e)}
            )
            return False, f"Error validating prompt: {str(e)}", None

    def validate_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Validate AI response against security rules.
        Returns (is_valid, error_message).
        """
        try:
            # Check for empty or too long responses
            if not response or not response.strip():
                return False, "Empty response", None
            
            if len(response) > self.max_tokens:
                return False, f"Response exceeds maximum length of {self.max_tokens} tokens", None

            # Check for PII using AWS Comprehend
            detected_entities, redacted_response = self.detect_pii(response)
            
            if detected_entities:
                entity_types = [entity['Type'] for entity in detected_entities]
                if self.redaction_enabled:
                    # Allow the response but with redacted PII
                    return True, f"PII detected and redacted: {', '.join(set(entity_types))}", redacted_response
                else:
                    # Block the response if redaction is disabled
                    return False, f"Found potential PII: {', '.join(set(entity_types))}", None

            # Check for blocked content
            blocked_content = self._check_blocked_content(response)
            if blocked_content:
                return False, f"Found blocked content: {', '.join(blocked_content)}", None

            return True, None, response

        except Exception as e:
            self.log_security_event(
                "response_validation_error",
                {"error": str(e)}
            )
            return False, f"Error validating response: {str(e)}", None

    def _check_blocked_content(self, text: str) -> List[str]:
        """Check for blocked content."""
        found_blocked = []
        text_lower = text.lower()
        
        # Check blocked keywords
        for keyword in self.content_filters['blocked_keywords']:
            if keyword.lower() in text_lower:
                found_blocked.append(keyword)

        # Check sensitive commands
        for command in self.content_filters['sensitive_commands']:
            if command.lower() in text_lower:
                found_blocked.append(command)

        return found_blocked

    def _check_sql_injection(self, text: str) -> bool:
        """Check for potential SQL injection attempts."""
        sql_patterns = [
            r'\bUNION\b.*\bSELECT\b',
            r';\s*DROP\s+TABLE',
            r';\s*DELETE\s+FROM',
            r';\s*INSERT\s+INTO',
            r'--\s*$',
            r'/\*.*\*/',
            r'\bOR\b.*=.*--'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_command_injection(self, text: str) -> bool:
        """Check for potential command injection attempts."""
        command_patterns = [
            r'`.*`',           # Backticks
            r'\$\(.*\)',       # Command substitution
            r';\s*rm\s+-rf',   # Dangerous file operations
            r'>\s*/dev/null',  # Output redirection
            r'\|\s*bash',      # Pipe to shell
            r'chmod\s+[0-7]{3,4}' # chmod commands
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def log_security_event(self, event_type: str, details: Dict):
        """Log security events for audit."""
        try:
            event = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'details': details
            }
            print(f"Security Event: {json.dumps(event, indent=2)}")
        except Exception as e:
            print(f"Error logging security event: {str(e)}")
            
        except Exception as e:
            print(f"Error logging security event: {str(e)}")