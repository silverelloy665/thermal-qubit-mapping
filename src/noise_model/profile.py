import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_thermal_profile(num_qubits: int, config: dict) -> dict:
    temps_mk = {}
    default_temp = config['thermal']['default_temp_mK']
    for q in range(num_qubits):
        temps_mk[q] = default_temp
        
    for hot in config['thermal']['hot_qubits']:
        idx = hot['index']
        if idx < num_qubits:
            temps_mk[idx] = hot['temp_mK']
            
    return temps_mk
