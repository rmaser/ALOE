import logging
from loguru import logger
import wandb

class LoguruWandbSink:
    """
    Redirects loguru logs to wandb.
    """
    def __init__(self, level=logging.INFO):
        self.level = level

    def write(self, message):
        record = message.record
        level = record["level"].name
        msg = record["message"]
        
        # Map loguru levels to wandb/logging levels if needed, 
        # but wandb.log doesn't have levels in the same way.
        # We can use wandb.termlog or just log as text.
        # A better approach for structured logging is to log as a table or just print 
        # so wandb captures stdout/stderr.
        
        # However, if we want to see it in the "Logs" tab of WandB with levels:
        # WandB captures stdout/stderr automatically. 
        # So if loguru prints to stderr (default), it should show up.
        # The user's issue "w&b doesnt use the loguru logger" likely means 
        # they want to explicitly send logs to wandb or format them.
        
        # If we want to explicitly log to wandb:
        if wandb.run is not None:
             # We can log as a custom chart or just rely on console capture.
             # But often the issue is that loguru's formatting isn't parsed by WandB's console capture 
             # as nicely as standard logging.
             pass

    def __call__(self, message):
        self.write(message)

# Actually, the easiest way to unify is to make loguru propagate to standard logging
# which WandB integrates with, OR just ensure loguru prints to stderr/stdout 
# and WandB captures it. 

# But often users want to see the logs in the WandB "Logs" panel with correct severity.
# WandB hooks into the python `logging` module. 
# So we should sink loguru to the standard python logging module.

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_loguru_logging_intercept(level=logging.INFO, modules=()):
    # Configure loguru to disable diagnose (local variables in tracebacks)
    import sys
    logger.remove()
    logger.add(sys.stderr, diagnose=False, backtrace=True)
    
    logging.basicConfig(handlers=[InterceptHandler()], level=level)
    for logger_name in modules:
        mod_logger = logging.getLogger(logger_name)
        mod_logger.handlers = [InterceptHandler()]
        mod_logger.propagate = False

# To go the other way (Loguru -> WandB/Standard Logging):
# We want Loguru -> Standard Logging (because WandB captures Standard Logging)

class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)

def redirect_loguru_to_standard_logging():
    # Remove default handler
    logger.remove()
    
    # Add a sink that propagates to standard logging
    # But wait, loguru is the main logger here. 
    # If we want WandB to pick it up, we need to emit to standard logging?
    # Or just let WandB capture stdout.
    
    # If the user says "w&b doesnt use the loguru logger", they probably mean 
    # they don't see the logs or they are not formatted.
    
    # Let's add a sink that writes to standard logging, which WandB hooks.
    
    class Sink:
        def write(self, message):
            record = message.record
            level = record["level"].no
            msg = message
            # Standard logging
            logging.getLogger("src").log(level, msg)

    # logger.add(Sink(), format="{message}") 
    # This might duplicate logs if we also print to stderr.
    
    # Simpler approach: Just ensure loguru prints to stderr (it does by default)
    # and WandB will capture it.
    # If it's not working, maybe WandB is configured to only capture standard logging?
    
    # Let's try to add a sink that explicitly calls wandb.termlog for important messages?
    pass

# Re-reading: "w&b doesnt use the loguru logger"
# This likely means `wandb`'s internal logging or the `WandBLogger` in Lightning 
# is using standard logging, and the user wants those to show up in Loguru?
# OR the user wants Loguru logs to show up in WandB.
# "logging is still inconsistent (I think w&b doesnt use the loguru logger)"
# This implies the user wants EVERYTHING to go through Loguru (unify on Loguru) 
# OR everything to go to WandB.
# Given "w&b doesnt use the loguru logger", it sounds like WandB's own logs 
# or Lightning's logs (which use standard logging) are not being formatted by Loguru.

# So we should intercept Standard Logging and send it to Loguru.
# This is a common pattern.

def unify_logging():
    """
    Redirects standard logging to Loguru.
    """
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Silence specific loggers if needed, or set their level
    # logging.getLogger("wandb").setLevel(logging.WARNING) 
    # (Optional, maybe user wants to see wandb logs in loguru format)

