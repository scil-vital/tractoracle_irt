import logging

root_logger = logging.getLogger('TO')

def add_logging_args(parser):
    parser.add_argument('--log_file', type=str, default=None,
                        help='File to log to. If not set, logs to stdout.')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Log level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.')
    return parser

def setup_logging(args):
    formatter = logging.Formatter(
        '%(levelname)s:%(filename)s: %(message)s')

    if args.log_file is not None:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(formatter)
        logging.basicConfig(filename=args.log_file,
                            handlers=[file_handler])
    elif args.log_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.basicConfig(handlers=[console_handler])

    root_logger.setLevel(args.log_level)

def setLevel(level):
    root_logger.setLevel(level)

def get_logger(name):
    return root_logger.getChild(name)
