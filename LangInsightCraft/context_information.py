from typing import List, Dict

class ContextualInformationExtraction:
    def __init__(self):
        pass

    def extract_entities(self, description: str) -> List[str]:
        """Extracts entities from the functional description."""
        # Example: Use a simple heuristic or an NLP model for NER
        entities = [word for word in description.split() if word.istitle()]
        return entities

    def extract_actions(self, description: str) -> List[str]:
        """Extracts actions (verbs) from the functional description."""
        actions = [word for word in description.split() if word.islower()]
        return actions

    def extract_context_scope(self, description: str, entities: List[str], actions: List[str]) -> str:
        """Determines the context scope based on entities and actions."""
        context_scope = f"The method likely performs {' and '.join(actions)} on entities like {' and '.join(entities)}."
        return context_scope

    def extract_contextual_info(self, description: str) -> Dict[str, List[str]]:
        """Extracts entities, actions, and context scope from the description."""
        entities = self.extract_entities(description)
        actions = self.extract_actions(description)
        context_scope = self.extract_context_scope(description, entities, actions)
        
        return {"Entities": entities, "Actions": actions, "Context Scope": context_scope}
