#!/usr/bin/env python3
"""Session Manager Tool for organizing counterfactual generation outputs."""

import os
import uuid
from datetime import datetime
from typing import Dict, Any
from smolagents import Tool

class SessionManagerTool(Tool):
    name = "session_manager"
    description = """Creates and manages a unique session directory for organizing all generated files.
    
    This tool creates a UUID-based session directory under 'statics/output_counterfactuals/sessions/' 
    and returns the session_id and full session_path for use by other tools.
    
    OUTPUT FORMAT: Returns a dictionary with:
    {
        "session_id": "unique_uuid_string",
        "session_path": "statics/output_counterfactuals/sessions/uuid_timestamp/",
        "timestamp": "YYYYMMDD_HHMMSS"
    }
    
    Use this at the start of analysis to create organized file storage.
    """
    
    inputs = {
        "create_session": {"type": "boolean", "description": "Set to True to create a new session directory", "nullable": True}
    }
    output_type = "object"
    
    def __init__(self):
        super().__init__()
        self.current_session = None
    
    def forward(self, create_session: bool = True) -> Dict[str, Any]:
        """Create a new session directory with UUID."""
        try:
            if create_session:
                # Generate unique session ID
                session_uuid = str(uuid.uuid4())[:8]  # First 8 chars for readability
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_id = f"{session_uuid}_{timestamp}"
                
                # Create session directory
                base_dir = "statics/output_counterfactuals/sessions"
                session_path = os.path.join(base_dir, session_id)
                
                os.makedirs(session_path, exist_ok=True)
                os.makedirs(os.path.join(session_path, "counterfactuals"), exist_ok=True)
                os.makedirs(os.path.join(session_path, "difference_maps"), exist_ok=True)
                os.makedirs(os.path.join(session_path, "transformed_inputs"), exist_ok=True)
                
                self.current_session = {
                    "session_id": session_id,
                    "session_path": session_path + "/",
                    "timestamp": timestamp
                }
                
                print(f"✅ Created session directory: {session_path}")
                return self.current_session
            else:
                if self.current_session:
                    return self.current_session
                else:
                    return {"error": "No active session. Set create_session=True to create one."}
                    
        except Exception as e:
            return {"error": f"Failed to create session: {str(e)}"} 