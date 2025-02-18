import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from peft import EigenLoRAConfig, get_peft_model, TaskType, load_peft_weights
from transformers import AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file, save_file
from utils import (
    combine_loras,
    get_eigenvectors,
    calculate_eigenloras,
    add_gs_vectors,
)


cola = load_peft_weights("ankit-vaidya19/cola_lora_r_8")
qnli = load_peft_weights("ankit-vaidya19/qnli_lora_r_8")

lora_dict = {}
lora_dict = combine_loras(lora_dict, cola, "cola")
lora_dict = combine_loras(lora_dict, qnli, "qnli")


eig_dict = get_eigenvectors(lora_dict, False)

mrpc_eigenlora = calculate_eigenloras(eig_dict, cola, 32, False)
stsb_eigenlora = calculate_eigenloras(eig_dict, cola, 32, False)


mrpc_eigenlora = add_gs_vectors(mrpc_eigenlora, 32)
stsb_eigenlora = add_gs_vectors(stsb_eigenlora, 32)


eigenlora_config = EigenLoRAConfig(
    r=8,
    num_components=64,
    task_type=TaskType.SEQ_CLS,
)


mrpc_config = AutoConfig.from_pretrained(
    "roberta-base",
    num_labels=2,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    config=mrpc_config,
)
model = get_peft_model(model, eigenlora_config)
model.save_pretrained(
    "./mrpc_eigenlora",
    save_eigenlora_components=True,
    save_eigenlora_loadings=True,
)
save_file(
    mrpc_eigenlora,
    "./mrpc_eigenlora/adapter_model.safetensors",
)


stsb_config = AutoConfig.from_pretrained(
    "roberta-base",
    num_labels=1,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    config=stsb_config,
)
model = get_peft_model(model, eigenlora_config)
model.save_pretrained(
    "./stsb_eigenlora",
    save_eigenlora_components=True,
    save_eigenlora_loadings=True,
)
save_file(
    stsb_eigenlora,
    "./stsb_eigenlora/adapter_model.safetensors",
)
