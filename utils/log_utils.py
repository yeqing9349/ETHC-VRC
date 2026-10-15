import csv
import os


class CsvLogger(object):
    """把一行行的指标追加写入CSV文件（续训时接着写），用于训练曲线和验证记录"""

    def __init__(self, path, fields):
        self.path = path
        self.fields = list(fields)
        if not os.path.isfile(path):
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(self.fields)

    def log(self, row):
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['' if row.get(k) is None else row.get(k) for k in self.fields])


def truncate_csv_log(path, last_epoch):
    """续训前只保留检查点所在 epoch 及之前的记录：中断前已写入、续训时会重新写入的行（包括写了一半的行）删去，
    训练曲线中不会出现重复的步数。从头训练时 last_epoch = -1，清空旧记录。返回删去的行数"""
    if not os.path.isfile(path):
        return 0
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    header, body = rows[0], rows[1:]
    col = header.index('epoch')

    def keep(r):
        try:
            return len(r) == len(header) and int(r[col]) <= last_epoch
        except ValueError:
            return False

    kept = [r for r in body if keep(r)]
    if len(kept) < len(body):
        tmp = path + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows([header] + kept)
        os.replace(tmp, path)
    return len(body) - len(kept)


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, violated_time_num=None, violated_node_num=None, route_length=None):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    if violated_time_num is not None:
        avg_violated_time_num = violated_time_num.mean().item()
        avg_violated_node_num = violated_node_num.mean().item()
        avg_route_length = route_length.mean().item()
        # Log values to screen
        print('epoch: {}, train_batch_id: {}, avg_cost: {}, avg_violated_time_num: {}, avg_violated_node_num: {}, avg_route_length: {}'.format(epoch, batch_id, avg_cost, avg_violated_time_num, avg_violated_node_num, avg_route_length))
    else:
        print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if tb_logger is not None:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if violated_time_num is not None:
            tb_logger.log_value('violated_time_num', violated_time_num.mean().item(), step)
            tb_logger.log_value('violated_node_num', violated_node_num.mean().item(), step)
            tb_logger.log_value('route_length', route_length.mean().item(), step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
