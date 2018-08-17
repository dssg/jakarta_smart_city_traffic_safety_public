# ============ Base imports ======================
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.pipeline.workers.pipeline_worker import PipelineWorker
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class GenericWorker(PipelineWorker):
    """Exemplary class which illustrates how to make a worker
    """
    def initialize(self, config_param1, config_param2, **kwargs):
        """this method gets called when the pipeline is first built, before individual worker processes have been spawned

        :param config_param1: example configuration parameter 1
        :param config_param2: example configuration parameter 2
        :param kwargs: additional keyword arguments
        :return: This should not return anything
        """
        # This method is passed all the parameters specified for this worker in the pipeline file
        # you should store these arguments so you can use them in the startup, run, and shutdown methods
        self.config_param1 = config_param1
        self.config_param2 = config_param2

    # this method gets called for each worker PROCESS after it has been spawned.
    # use it to set environment variables or set up connections to the GPU, etc.
    def startup(self):
        """this method gets called for each worker PROCESS after it has been spawned.

        # use it to set environment variables or set up connections to the GPU, etc.
        :return: Nothing
        """
        pass

    def run(self, item, *args, **kwargs):
        """Function which processes a single item (frame) in the pipeline

        Note: If the pipeline does not define a previous task for a worker, then the run function is only called once
        and is not passed anything.
        This allows, for example, for a worker to loop through all files in a directory and place items into the output
        queue.
        If the pipeline DOES define a previous task for a worker, then the run function is called once for each item in
        the input queue.

        When this function is done with each item, you should call self.done_with_item(item) to place
        it in the output queue

        :param item: queue item from the input queue (in our case this is a frame dictionary
        :param args: additional position argumants (currently unused)
        :param kwargs: additional keyword arguments (currently unused)
        :return: should not return anything
        """
        self.logger.info("This is an example of a logger info statement")
        self.logger.debug("This statement will only show in the terminal when running in debug mode")
        self.logger.debug("However, debug statements always get written to the output log file")
        # be sure to call either this method or self.done_with_items(items) to feed the item into the next queue(s)
        self.done_with_item(item)
        # if there is no worker after this one, then you should not call self.done_with_item

    # This method gets called after the queue which this worker monitors has been shut down
    def shutdown(self):
        """ last function to be called before the worker is shut down

        This function is called once when the worker is being shut down, after there are no more items to be
        processed in the input queue.

        :return: should not return anything
        """
        # you can reference other config parameters anywhere in this file:
        self.logger.debug(f"The value of conf.paths.output is {conf.paths.output}")

    #==============================
    #= Support Functions/Classes ==
    #==============================
    def example_instance_method(self, arg1):
        """ Example for an additional helper function with access to instance

        # this method has a reference to the calling class, so you can access instance variables:
        # call this method using self.example_instance_method

        :param arg1: example argument
        :return: return stuff
        """
        needed_param = self.config_param1

    @staticmethod
    def example_static_method(arg1):
        """ Example for an additional helper function without access to instance

        # this method does not have a reference to the calling class
        # call this method using GenericWorker.example_static_method

        :param arg1:
        :return:
        """
        pass

