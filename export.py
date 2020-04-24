import torch
import argparse
import pickle

def extract_keys(filename, keys):
    data = torch.load(filename)
    export_dict = {}
    for key1 in keys:
        export_dict[key1] = {}
        for key2 in data[key1].keys():
            m = data[key1][key2]
            export_dict[key1][key2] = m.numpy()
    return export_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model')
    parser.add_argument('--osi')

    args = parser.parse_args()

    d = extract_keys(args.model, ['actor'])
    if args.osi is not None:
        d.update(extract_keys(args.osi, ['net']))
    with open('model.pkl', 'wb') as f:
        pickle.dump(d, f)
