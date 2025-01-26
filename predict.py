import argparse
import torch
import Datasetutils
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model"
)
parser.add_argument(
    "--data", type=str, default="./data/glue-sst2", help="location of the data corpus"
)
parser.add_argument(
    "--model",
    type=str,
    default="LSTM",
    help="type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)",
)
parser.add_argument("--emsize", type=int, default=400, help="size of word embeddings")
parser.add_argument(
    "--nhid", type=int, default=200, help="number of hidden units per layer"
)
parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,  # 你可能需要调整它
    help="initial learning rate",
)
parser.add_argument("--clip", type=float, default=0.15, help="gradient clipping")
parser.add_argument("--epochs", type=int, default=10, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="N", help="batch size"
)
parser.add_argument("--bptt", type=int, default=35, help="sequence length")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.05,
    help="dropout applied to layers (0 = no dropout)",
)
parser.add_argument(
    "--tied", action="store_true", help="tie the word embedding and softmax weights"
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--log-interval", type=int, default=50, metavar="N", help="report interval"
)
parser.add_argument(
    "--save", type=str, default="model.pt", help="path to save the final model"
)
parser.add_argument(
    "--nhead",
    type=int,
    default=2,
    help="the number of heads in the encoder/decoder of the transformer model",
)
parser.add_argument(
    "--dry-run", action="store_true", help="verify the code and the model"
)
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

test_dataset = Datasetutils.Mydataset(args.data, args.bptt, "test")
test_dataloader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
)
model = torch.load("model.pt")
print(model.to(device))


def make_sentence(input):
    seq = ""
    for num in input:
        seq += Datasetutils.ALL_DICT.idx2word[num] + " "
    return seq


# model.eval()
# with torch.no_grad():
#     for i in range(50):
#         data, label = test_dataset[i]
#         data = data.to(device)
#         label = label.to(device)
#         data = data.unsqueeze(0)
#         data = data.permute(1, 0)
#         outputs = model(data)
#         prediction = torch.argmax(outputs, dim=1)

#         print("sequence:", make_sentence(data.permute(1,0).squeeze(0)),"| predict:",prediction,"| label:",label.item())
lossF = torch.nn.CrossEntropyLoss()
model.eval()
with torch.no_grad():
    correct, totalLoss = 0, 0
    for data, label in test_dataloader:
        data = data.to(device)
        label = label.to(device)
        data = data.permute(1,0)
        output = model(data)
        loss = lossF(output,label)
        predictions = torch.argmax(output, dim=1)
        totalLoss += loss
        correct += torch.sum(predictions == label)
    Acc = correct / (args.batch_size * len(test_dataloader))
    Loss = totalLoss / len(test_dataloader)
    print("Acc: ",Acc.item(),"| Loss: ",totalLoss.item())