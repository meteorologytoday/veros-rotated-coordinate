from tqdm import tqdm
from veros import runtime_settings
print("Setting veros.runtime_settings...")
setattr(runtime_settings, "backend", "numpy")
setattr(runtime_settings, "force_overwrite", True)
setattr(runtime_settings, 'device', 'cpu')
from veros_case_setup import VerosCaseSetup

total_time = 86400 * 10
ocn_model = VerosCaseSetup()

print("Setup ocean model")
ocn_model.setup()
settings = ocn_model.state.settings

print("Step ocean model")
total_steps = int(total_time / settings.dt_tracer )
for step in tqdm(range(total_steps)):
    ocn_model.step(ocn_model.state)

