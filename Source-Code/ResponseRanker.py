class ResponseRanker:
    def __init__(self, reward_model, alaro):
        self.reward_model = reward_model
        self.alaro = alaro

    def rank_responses(self, am, suggested_methods):
        rewards = []
        for sm in suggested_methods:
            sim_score = self.reward_model.semantic_similarity(am, sm)
            edit_score = self.reward_model.edit_distance(am, sm)
            R = self.reward_model.reward_score(am, sm, self.alaro.w_sim, self.alaro.w_edit)
            self.alaro.update_weights(R, sim_score, edit_score)
            rewards.append((sm, R))
        
        rewards.sort(key=lambda x: x[1], reverse=True)
        return rewards

    def generate_feedback_prompt(self, am, ranked_methods):
        ranked_names = " > ".join([sm for sm, _ in ranked_methods])
        feedback_prompt = (f"Based on the semantic similarity and edit distance scores of the suggested method names "
                           f"compared to the actual method name: {am}, the ranking of suggested method names would be "
                           f"as follows: {ranked_names}")
        return feedback_prompt

# Example usage
if __name__ == "__main__":
    # Example data
    actual_method_name = "removeItemListener"
    suggested_method_names = [
        "deleteItemListener",
        "detachItemListener",
        "eraseItemListener",
        "removeItemHandler"
    ]

    # Initialize models
    reward_model = RewardModel()
    alaro = ALARO()
    ranker = ResponseRanker(reward_model, alaro)

    # Rank responses
    ranked_methods = ranker.rank_responses(actual_method_name, suggested_method_names)

    # Generate feedback prompt
    feedback_prompt = ranker.generate_feedback_prompt(actual_method_name, ranked_methods)
    print(feedback_prompt)
