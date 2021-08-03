import time
import json

from ax.modelbridge import get_sobol, get_GPEI
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient
from framework_problem import ContextualRunner, Agent, arrival_aggregate_reward

########
# Define problem
NUM_USER = 3
NUM_CREATOR = 3
num_trials = 75
NUM_RUNS = 400
Lambda_list = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]  # lambda_list=[lambda_i, lambda_j]
new_lambda = [[0, 0, 0], [0, 0, 0]]
seg_i = [1 / 3] * 3
seg_j = [1 / 3] * 3  # uniform distribution
theta_i = [0.5, 0.5, 0.5]
theta_j = [0.5, 0.5, 0.5]
# Context_List=[seg_i, seg_j, lambda_i, lambda_j]
Lambda_list = new_lambda
Context_List = [[1 / 3] * 3, [1 / 3] * 3, [1] * 3, [1] * 3]
num_contexts = 9


while new_lambda <= Lambda_list:
    Lambda_list = new_lambda
    k = [seg_i, seg_j]
    Context_List = k.extend(Lambda_list)
    num_contexts = 9
    benchmark_problem = ContextualRunner(
        num_contexts=num_contexts,
        context_list=Context_List
    )


    gs = GenerationStrategy(
        name="GPEI",
        steps=[
            GenerationStep(get_sobol, 8),
            GenerationStep(get_GPEI, -1),
        ],
    )
    axc = AxClient(generation_strategy=gs)

    experiment_parameters = benchmark_problem.contextual_parameters()
    axc.create_experiment(
        name="aggregated_reward_experiment",
        parameters=experiment_parameters,
        objective_name="aggregated_reward",
        minimize=False,
        parameter_constraints=["rho_00 + rho_01 +rho_02 = 1",
                               "rho_10 + rho_11 +rho_12 = 1",
                               "rho_20 + rho_21 +rho_22 = 1"],
        overwrite_existing_experiment=True,
    )

    def evaluation_aggregated_reward(parameters):
        x = []
        for value in parameters.values():
            x.append(value)
        aggregated_reward = benchmark_problem.f(x, theta_i, theta_j)
        return {"aggregated_reward": (aggregated_reward, 0.0)}


    for itrial in range(num_trials):
        parameters, trial_index = axc.get_next_trial()
        aggregated_res = evaluation_aggregated_reward(parameters)
        axc.complete_trial(trial_index=trial_index, raw_data=aggregated_res)

    best_parameters = axc.get_best_parameters()
    Rho_ij = []
    for value in best_parameters.values():
        Rho_ij.append(value)

    new_agent = Agent(seg_i, seg_j, Lambda_list, Rho_ij)
    new_lambda = []
    for i in range(NUM_USER + 1):
        new_lambda.append(new_agent.user_arrival_rate(i, theta_i))
    for j in range(NUM_CREATOR + 1):
        new_lambda.append(new_agent.creator_arrival_rate(j, theta_j))
