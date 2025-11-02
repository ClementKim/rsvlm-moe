import numpy as np
import argparse, torch, random

from rsvqa import rsvqa_main
from baseline_models import initialize_model
    
def str_to_bool(s):
    if s.lower() == 'true':
        return True
    
    elif s.lower() == 'false':
        return False
    
    else:
        raise ValueError("Boolean value expected.")
    
def str_to_int(s):
    try:
        return int(s)
    
    except ValueError:
        return None

def main(args):
    # control randomness
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.param = str_to_int(args.param)
    args.train = str_to_bool(args.train)
    args.val = str_to_bool(args.val)
    args.test = str_to_bool(args.test)

    vlm = initialize_model(args)

    if args.dataset == "rsvqa":
        rsvqa_main(args, vlm, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "qwen")
    parser.add_argument("--param", type = str, default = "3")
    parser.add_argument("--batch", type = int, default = 16)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--dataset", type = str, default = "rsvqa")
    parser.add_argument("--train", type = str, default = "False")
    parser.add_argument("--val", type = str, default = "False")
    parser.add_argument("--test", type = str, default = "True")
    args = parser.parse_args()

    main(args)