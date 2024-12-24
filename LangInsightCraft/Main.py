from lang_insight_craft import LangInsightCraft

if __name__ == "__main__":
    # Example usage
    input_description = "此方法核算发票金额，考虑税率和折扣。"  # Example description
    csv_file_path = 'java_train.csv'  # Path to the CSV containing functional descriptions and method names

    # Initialize LangInsightCraft and create the context-enriched prompt
    lang_insight_craft = LangInsightCraft(csv_file_path)
    context_enriched_prompt = lang_insight_craft.create_context_enriched_prompt(input_description)

    # Now, you can pass the `context_enriched_prompt` to LLMs for method name generation.
    print(context_enriched_prompt)
