import time
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix

def model_train(training_configs, train_loader, model, loss_func, optimizer, epoch, logger):
    device = training_configs['train']['device']
    max_epoch = training_configs['train']['max_epochs']

    model.train()
    tmp_loss = 0.
    for step, (data, label) in enumerate(
            train_loader):
        if (len(data.shape) == 3 and training_configs['model']['type'] == '2d'
                and training_configs['model']['name'] not in ['FBMSNet']):
            data = data.unsqueeze(1)
        elif len(data.shape) == 4 and training_configs['model']['type'] == '1d':
            data = data.squeeze(1)
        data, label = data.to(device), label.to(device)
        label = torch.squeeze(label)
        label = torch.as_tensor(label, dtype=torch.long, device=device)

        output = model(data)

        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tmp_loss += loss.item() * data.size(0)

    epoch_loss = tmp_loss / len(train_loader.dataset)
    logger.info(
        'Epoch:[{}/{}]\t| train loss: {:.5f}'.format(epoch + 1, max_epoch, epoch_loss))


def model_test(training_configs, test_loader, model, epoch, logger):
    if epoch == 900:
        start = time.perf_counter()

    device = training_configs['train']['device']
    max_epoch = training_configs['train']['max_epochs']

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, label in test_loader:
            if (len(data.shape) == 3 and training_configs['model']['type'] == '2d'
                    and training_configs['model']['name'] not in ['FBMSNet']):
                data = data.unsqueeze(1)
            elif len(data.shape) == 4 and training_configs['model']['type'] == '1d':
                data = data.squeeze(1)
            test_data = data.type(torch.FloatTensor).to(device)
            test_label = model(test_data).float().cpu()
            predict_y = torch.max(test_label, dim=1)[1]

            label = torch.squeeze(label)
            label = label.long()
            all_predictions.extend(predict_y.numpy())
            all_labels.extend(label.cpu().numpy())

    epoch_acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    logger.info(f'Epoch:[{epoch+1}/{max_epoch}]\t| Test Accuracy: {epoch_acc:.5f}\t| Precision: {precision:.5f}\t| Recall: {recall:.5f}\t| F1 Score: {f1:.5f}\t| Kappa: {kappa:.5f}')
    logger.info(f'\nall_labels:\n{all_labels}\nall_predictions:\n{all_predictions}')
    logger.info(f'Confusion Matrix:\n{conf_matrix}')

    if epoch == 900:
        end = time.perf_counter()
        execution_time_ms = (end - start) * 1000
        print("-" * 60)
        print(f"print 执行时间: {execution_time_ms:.4f} 毫秒")
        print("-" * 60)

    return epoch_acc

def accuracy_topk(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
