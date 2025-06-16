import anthropic
import yaml
import pdb

evaluator_prompt = """
You are a helpful assistant that evaluates the quality of a product purchase.

You will be given a product that has been purchased by a shopping assistant, and a criteria for the product purchase.
You need to provide a binary score (0 or 1) based on whether the purchased product meets the criteria. If the criteria is very specific (e.g. a t-shirt that is a shade/pattern of black specific to a product), and the product is similar enough (e.g. a t-shirt that is also a shade/pattern of black), you should give a score of 1.

The purchased product:
Product name: {product_name}
Product description: {product_description}
Product price: {product_price}
Product attributes: {product_attributes}
Product features description: {product_features_description}
Selected options when purchasing: {selected_options}

The criteria for the product purchase:
CRITERIA

Your output should be a single token, either "0" or "1".
"""

cost_dict = {
    "claude-3-7-sonnet-latest": {"input": 3/10**6, "output": 15/10**6},
}

class WebShopLMEvaluator:

    def __init__(self, model_name: str, args: dict):
        self.model_name = model_name
        self.args = args
        secrets = yaml.safe_load(open(args.secrets_file, "r"))
        self.anthropic_client = anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
        self.llm_cost = 0
        self.num_evaluations = 0

    def evaluate(self, purchased_product: dict, goal: dict) -> dict:
        product_specific_prompt = evaluator_prompt.format(
            product_name=purchased_product['product_name'],
            product_description=purchased_product['description'] if 'description' in purchased_product else "N/A",
            product_price=purchased_product['price'],
            product_attributes=', '.join(purchased_product['attributes']),
            product_features_description="\n- ".join(purchased_product['bullet_points']) if 'bullet_points' in purchased_product else "N/A",
            selected_options=', '.join([f"{k}: {v}" for k, v in purchased_product['options'].items()]) if purchased_product['options'] else 'N/A',
        )

        # Evaluate if product is the right type
        prompt = product_specific_prompt.replace("CRITERIA", f"The product is of the type: {goal['product_type']}")
        r_type = self.get_score(prompt)

        num_attr_matches = 0
        attr_scores = {}
        for attr in goal['attributes']:
            prompt = product_specific_prompt.replace("CRITERIA", f"The product has the attribute: {attr}")
            attr_scores[attr] = self.get_score(prompt)
            num_attr_matches += attr_scores[attr]
        
        # Evaluate if product is the right price
        prompt = product_specific_prompt.replace("CRITERIA", f"The product is priced under ${goal['price_upper']}.")
        r_price = self.get_score(prompt)
        
        option_scores = {}
        num_option_matches = 0
        for k, v in goal['goal_options_dict'].items():
            prompt = product_specific_prompt.replace("CRITERIA", f"The product's {k} property is {v}")
            option_scores[k] = self.get_score(prompt)
            num_option_matches += option_scores[k]

        total_reward = r_type * (num_attr_matches + num_option_matches + r_price) / (len(goal['attributes']) + len(goal['goal_options_dict']) + 1)
        self.num_evaluations += 1
        return {
            'r_type': r_type,
            'attr_scores': attr_scores,
            'r_attr': num_attr_matches / len(goal['attributes']),
            'option_scores': option_scores,
            'r_option': num_option_matches / len(goal['goal_options_dict']),
            'r_price': r_price,
            'total_reward': total_reward,
        }

    def get_score(self, prompt: str) -> int:
        num_tries = 0
        while True:
            try:
                response = self.anthropic_client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                response = response.content[0].text.strip()
                cost = cost_dict[self.model_name]["input"] + cost_dict[self.model_name]["output"] * len(response)
                self.llm_cost += cost
                assert '1' in response or '0' in response
                assert not ('1' in response and '0' in response)
            except Exception as e:
                num_tries += 1
                print(f"Error: {e}")
                if num_tries > 3:
                    pdb.set_trace()
                continue

            return int(response)