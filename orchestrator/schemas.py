"""
Orchestrator Schemas

This module contains Pydantic models for structured output from LLMs.
These schemas ensure reliable JSON parsing and validation.
"""

import json
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, computed_field
from pydantic.config import ConfigDict


class ArgumentKV(BaseModel):
    """Key-value pair for tool arguments that satisfies Azure OpenAI's additionalProperties requirements."""
    model_config = ConfigDict(extra='forbid')
    
    key: str = Field(description="Argument name")
    value_json: str = Field(
        description="JSON-encoded value for this argument (can be string, number, boolean, object, or array)"
    )


class ToolSelection(BaseModel):
    """Schema for tool selection responses from LLMs, compatible with Azure OpenAI structured output."""
    model_config = ConfigDict(
        extra='forbid',  # Ensures additionalProperties=false at root level
        json_schema_extra={
            "example": {
                "tool_name": "get_weather",
                "server": "weather_service", 
                "arguments_kv": [
                    {"key": "location", "value_json": "\"Tokyo\""},
                    {"key": "units", "value_json": "\"metric\""}
                ],
                "reasoning": "Selected weather tool to get current conditions for the requested location"
            }
        }
    )
    
    tool_name: Optional[str] = Field(
        description="Name of the selected tool, or null if no suitable tool found"
    )
    
    server: Optional[str] = Field(
        description="Name of the server containing the tool, or null if no tool selected"
    )
    
    arguments_kv: List[ArgumentKV] = Field(
        default_factory=list,
        description="Arguments as key-value pairs with JSON-encoded values to satisfy Azure schema requirements"
    )
    
    reasoning: str = Field(
        description="Brief explanation of why this tool was selected or why no tool was suitable"
    )

    @computed_field  # type: ignore[misc]
    @property
    def arguments(self) -> Dict[str, Any]:
        """Backwards compatibility: convert arguments_kv to a dictionary."""
        result: Dict[str, Any] = {}
        for kv in self.arguments_kv:
            try:
                # Try to parse as JSON first
                result[kv.key] = json.loads(kv.value_json)
            except (json.JSONDecodeError, ValueError):
                # Fallback: treat as string if not valid JSON
                result[kv.key] = kv.value_json
        return result

class QueryDecomposition(BaseModel):
    """Schema for query decomposition responses from LLMs."""
    model_config = ConfigDict(
        extra='forbid',  # Ensures additionalProperties=false for Azure OpenAI compatibility
        json_schema_extra={
            "example": {
                "sub_queries": [
                    "What is the weather in Tokyo?",
                    "What activities are popular in Tokyo during this weather?",
                    "What should I pack for Tokyo weather?"
                ]
            }
        }
    )
    
    sub_queries: list[str] = Field(
        description="List of sub-queries that decompose the original query into smaller, actionable parts"
    )

# Future schemas can be added here as needed
# class QueryClassification(BaseModel):
#     """Schema for query classification responses from LLMs."""
#     model_config = ConfigDict(extra='forbid')  # Azure OpenAI compliance
#     
#     classification: str = Field(description="Either 'TOOLS' or 'CONVERSATIONAL'")
#     confidence: float = Field(description="Confidence score between 0 and 1")
#     reasoning: str = Field(description="Explanation for the classification")
