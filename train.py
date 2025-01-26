import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# 最后画图展示损失变化
history = {"Valid Loss": [], "Valid Accuracy": []}
# 全局最佳验证损失
Best_val_loss = None

def TrainNet(
    epochs: int,
    batch_size: int,
    net: torch.nn.Module,
    trainDataLoader: torch.utils.data.dataloader.DataLoader,
    validDataLoader: torch.utils.data.dataloader.DataLoader,
    device: str,
    lossF: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
):
    """使用tqdm可视化训练进度的训练函数,一个epoch完会保存模型为model.pt
    Args:
        epochs (int): 训练的轮数
        batch_size (int): batch大小
        net (torch.nn.Module): 网络模型
        trainDataLoader (torch.utils.data.dataloader.DataLoader): 训练集的dataloader
        validDataLoader (torch.utils.data.dataloader.DataLoader): 验证集的dataloader
        device (str): 训练设备
        lossF (torch.nn.modules.loss._WeightedLoss): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
    """    
    for epoch in range(1, epochs + 1):
        processBar = tqdm(trainDataLoader, unit="step")
        net.train(True)
        for step, (data, labels) in enumerate(processBar):
            # 取出数据后转移到device上
            data = data.to(device)
            labels = labels.to(device)
            # 清空梯度
            net.zero_grad()
            # 维度变换，lstm输入的数据若没有设置batchfirst则应该是[seq_len,batch_size]的shape
            # 详情可见：https://www.jianshu.com/p/3464278fcf2d
            # 所以这里进行变换，变换为[seq_len,batch_size]的shape
            data = data.permute(1,0)
            outputs = net(data)
            loss = lossF(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels) / labels.shape[0]
            loss.backward()
            optimizer.step()

            processBar.set_description(
                "[%d/%d] Loss: %.4f, Acc: %.4f"
                % (epoch, epochs, loss.item(), accuracy.item())
            )
            # 进行验证集的验证
            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)
                for validdata, labels in validDataLoader:
                    validdata = validdata.to(device)
                    labels = labels.to(device)
                    validdata = validdata.permute(1,0)
                    outputs = net(validdata)
                    loss = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)

                    totalLoss += loss
                    correct += torch.sum(predictions == labels)
                validAccuracy = correct / (batch_size * len(validDataLoader))
                validLoss = totalLoss / len(validDataLoader)
                history["Valid Loss"].append(validLoss.item())
                history["Valid Accuracy"].append(validAccuracy.item())
                processBar.set_description(
                    "[%d/%d] Loss: %.4f, Acc: %.4f, Valid Loss: %.4f, Valid Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        loss.item(),
                        accuracy.item(),
                        validLoss.item(),
                        validAccuracy.item(),
                    )
                )
                # 设置使用全局变量
                global Best_val_loss
                # 进行更新保存最佳损失的模型
                if not Best_val_loss or validLoss < Best_val_loss:
                    with open("model.pt", 'wb') as f:
                        torch.save(net, f)
                    Best_val_loss = validLoss
        processBar.close()

    plt.plot(history['Valid Loss'],label = 'Valid Loss')
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # 对验证集准确率进行可视化
    plt.plot(history["Valid Accuracy"], color="red", label="Valid Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
