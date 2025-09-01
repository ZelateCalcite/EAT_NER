from argparse import ArgumentParser


def parse_train_config():
    parser = ArgumentParser(description='Train Arguments')
    parser.add_argument('--epoch', default=25, type=int, help='')
    parser.add_argument('--trd', type=str, help='train data')
    parser.add_argument('--evd', default='', type=str, help='eval data')
    parser.add_argument('--ted', default='', type=str, help='test data')
    parser.add_argument('--eval', default=False, type=bool, help='do eval')
    parser.add_argument('--ul', default=False, type=bool, help='use lora')
    parser.add_argument('--cp', type=str, help='checkpoint')
    parser.add_argument('--lsd', default='', type=str, help='load state dict')
    parser.add_argument('--on', type=str, help='output state dict name')

    return parser.parse_args()


if __name__ == '__main__':
    pass
