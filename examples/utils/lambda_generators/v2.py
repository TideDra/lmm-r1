import random
def lambda_generator(experience_maker):
    if random.random() < 0.5:
        lambda_a = 1.0
    else:
        lambda_a = random.uniform(0.0,1.0)
    return {"acceptance_ensemble_lambda":lambda_a,"distribution_ensemble_lambda":1.0}