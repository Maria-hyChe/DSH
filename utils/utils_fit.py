import os

import torch
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score, metrics


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, colorDict_GRAY,local_rank=0):
    total_loss = 0
    total_accuracy = 0
    total_kappa = 0
    total_precision = 0
    total_recall = 0
    total_auc = 0
    total_f_score = 0

    val_loss = 0
    val_accuracy = 0
    val_kappa = 0
    val_precision = 0
    val_recall = 0
    val_auc = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train\n')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                overall_accuracy, kappa, precision, recall, f1, cl_acc, auc = metrics(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    overall_accuracy, kappa, precision, recall, f1, cl_acc, auc = metrics(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_accuracy += overall_accuracy
        total_kappa += kappa
        total_precision += precision
        total_recall += recall
        total_auc += auc
        total_f_score += f1

            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'auc': total_auc / (iteration + 1),
                                'acc': total_accuracy / (iteration + 1),
                                'precis': total_precision / (iteration + 1),
                                'recall': total_recall / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'kappa': total_kappa / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs     = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice

            overall_accuracy, kappa, precision, recall, f1, cl_acc, auc = metrics(outputs, labels)
            val_loss += loss.item()
            val_accuracy += overall_accuracy
            val_kappa += kappa
            val_precision += precision
            val_recall += recall
            val_auc += auc
            val_f_score += f1
            
            if local_rank == 0:
                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                    'auc': val_auc / (iteration + 1),
                                    'acc': val_accuracy / (iteration + 1),
                                    'precis': val_precision / (iteration + 1),
                                    'recall': val_recall / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'kappa': val_kappa / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        train_metric = [total_accuracy / epoch_step, total_auc / epoch_step, total_precision / epoch_step,
                        total_recall / epoch_step, total_kappa / epoch_step, total_f_score / epoch_step]
        val_metric = [val_accuracy / epoch_step_val, val_auc / epoch_step_val, val_precision / epoch_step_val,
                      val_recall / epoch_step_val, val_kappa / epoch_step_val, val_f_score / epoch_step_val]

        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        loss_history.append_metric(epoch + 1, train_metric, val_metric)
        eval_callback.on_epoch_end(epoch + 1, model_train, colorDict_GRAY)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))