import torchvision.models as models
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import argparse
import copy
import os

CORRUPTIONS = ['defocus_blur','glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog','brightness']

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0, help='number of data loader threads per dataloader. 0 will use the main thread and is good for debugging')
parser.add_argument("--num_bn_updates", type=int, default=10, help="number of batch norm updates")
parser.add_argument("--corruption", type=str, default ="glass_blur")
parser.add_argument("--severity", type=int, default=1,)
parser.add_argument("--apply_bn", action="store_true")
parser.add_argument("--evaluate",action="store_true", help="run a full validation on all corruptions and all severities")
parser.add_argument("--clean_eval",action="store_true", help ="validate the model on the clean uncorrutped dataset" )
args = parser.parse_args()


def validate(model, val_loader,device):
    # Validate the model on a given dataset

    if args.apply_bn:
        model = update_bn_params(model, val_loader, args)

    model.eval()
    with torch.no_grad():
        total_loss, total_acc, num_batches = 0., 0., 0
        for batch_idx, (data, label) in enumerate(val_loader):
            num_batches += 1
            # move data to our device
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output,label)
            pred = torch.argmax(output, dim=1)
            correct = torch.sum(pred == label)
            acc = correct / len(data)
            total_loss += loss.item()
            total_acc += acc.item()

            # log the loss
            if batch_idx % 100 == 0:
                print(f"Validation step {batch_idx}/{len(val_loader)}")
    # normalize accuracy and loss over the number of batches
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    #print(f"Validation complete. Loss {avg_loss:.6f} accuracy {avg_acc:.2%}")

    return avg_loss, avg_acc

def validate_c(model,val_transform,device):

    # validate the model on a given corrupted dataset

    print(f"{args.corruption} severity {args.severity}")
    valdir = os.path.join("/project/cv-ws2122/shared-data1/data/dataset_folder", args.corruption, str(args.severity))
    val_dataset = datasets.ImageFolder(valdir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    loss, acc1 = validate(model,val_loader,device)
    print(f"Validation complete. Loss {loss:.6f} accuracy {acc1:.2%}")

    return

def validate_all_c(model,val_transform,device):
    # validate the model on the full corrupted dataset
    corruption_accs = {}
    for c in CORRUPTIONS:
        print(c)
        for s in range(1, 4):
            valdir = os.path.join("/project/cv-ws2122/shared-data1/data/dataset_folder", c, str(s))
            val_dataset = datasets.ImageFolder(valdir, transform=val_transform)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            loss, acc1 = validate(model,val_loader,device)
            if c in corruption_accs:
                corruption_accs[c].append(acc1*100)
            else:
                corruption_accs[c] = [acc1*100]
            print(f"Validation of {c} severity {s} complete. Loss {loss:.6f} accuracy {acc1:.2%}")


    return corruption_accs



def update_bn_params(model, val_loader, args):
    # START TODO #################

    val_loader = torch.utils.data.DataLoader(val_loader.dataset,
                                             batch_size=val_loader.batch_size,
                                             shuffle=True, num_workers=val_loader.num_workers)
    #1. make a copy of the model before evaluating on a new corruptuon type
    # see copy.deepcopy
    model = copy.deepcopy(model)
    model.eval()


    #2. Switch only BN layers to training mode.
    #module.train() will enable training mode
    #See: https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=apply#torch.nn.Module.apply
    #to apply your switching function to multiple layers

    print("Updating BN params (num updates:{})".format(args.num_bn_updates))

    #3. Switch off grad computation since we don't need it

    #4 Loop over test samples and stop if args.num_bn_updates has reached
    # If BN layers are in training mode, the model will accumulate statistics

    print("Done.")

    # END TODO ##################
    return model




def main():

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running training on: {device}")

    # use pre trained resnet18
    resnet18 = models.resnet18(pretrained=True)
    model = resnet18

    # send model to device
    model = model.to(device)

    # define transformation
    val_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Evaluation on the uncorrupted clean dataset
    if args.clean_eval:
        print("clean validation")
        val_dataset  = datasets.ImageFolder("/project/cv-ws2122/shared-data1/data/imagenet200", transform=val_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        loss, acc1 = validate(model,val_loader,device)
        print(f"Validation complete. Loss {loss:.6f} accuracy {acc1:.2%}")

    elif args.evaluate:
        # validate all corruptions

        corruption_accs = validate_all_c(model,val_transform,device)
        print(corruption_accs)


    else:
        # validate on curroption given by user or default
        validate_c(model,val_transform,device)



if __name__ == '__main__':
    main()
