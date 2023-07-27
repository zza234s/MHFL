import os
import sys

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner

from federatedscope.contrib.common_utils import result_to_csv, plot_num_of_samples_per_classes, \
    show_per_client_best_individual

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    print(os.getcwd())
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)

    if init_cfg.show_label_distribution:
        plot_num_of_samples_per_classes(data, modified_cfg)  # 是否可视化train/test dataset的标签分布

    init_cfg.freeze(inform=False)  # TODO:添加是否显示主cfg详细配置的变量
    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=client_cfgs)
    _ = runner.run()

    # show result
    client_summarized_test_acc = _['client_summarized_avg']['test_acc']
    client_summarized_weighted_avg = _['client_summarized_weighted_avg']['test_acc']
    print(f'client_summarized_avg_test_acc:{client_summarized_test_acc}')  # acc求平均
    print(f'client_summarized_weighted_avg_test_acc:{client_summarized_weighted_avg}')  # 加权平均acc
    best_round = runner.server.best_round
    result_to_csv(_, init_cfg, best_round,runner)

