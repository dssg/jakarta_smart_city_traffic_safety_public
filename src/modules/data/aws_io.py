# ============ Base imports ======================
import os
# ====== External package imports ================
import boto3
import botocore
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
logging.getLogger('boto').propagate = False  # prevents boto3 logger messages
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class AwsIo:
    """Manages all interaction with AWS

    enables copying of video files to/from database, getting video sizes, and checking if a video exists.
    uses credentials found in the configuration files

    """
    def __init__(self):
        """Load credentials from file and store in this class instance

        """
        self.s3_conf = conf.aws.s3
        # create transfer object that handles multithreading, failed downloads, etc. automatically (supposedly)
        self.client = boto3.client('s3', 'us-west-2', aws_access_key_id=self.s3_conf.access_key,
                                   aws_secret_access_key=self.s3_conf.secret_access_key)
        self.transfer = boto3.s3.transfer.S3Transfer(self.client)
        # get s3 resource for other io stuff that transfer can't do
        self.s3 = boto3.resource("s3", aws_access_key_id=self.s3_conf.access_key,
                                 aws_secret_access_key=self.s3_conf.secret_access_key)
        self.bucket = self.s3.Bucket(conf.aws.s3.bucket)

    def vid_copy_file_to_s3(self, filepath):
        """copy a video file from filepath to the listed bucket, keeping the same name

        :param filepath: location on the local file system to copy the video from
        """
        filename = os.path.basename(filepath)
        self.transfer.upload_file(filepath, bucket=self.bucket.name, key=os.path.join(conf.dirs.s3_videos, filename))

    def vid_copy_s3_to_file(self, filename, dirname):
        """ copy a video from an s3 bucket to a file location

        :param filename: name of file on s3
        :param dirname: directory to which to copy the file
        """
        self.transfer.download_file(filename=os.path.join(dirname, filename), bucket=self.bucket.name, key=os.path.join(conf.dirs.s3_videos, filename))

    def vid_copy_s3_to_obj(self, filename):
        """copy a video from an s3 bucket to a data stream

        :param filename: name of file on s3
        :return: raw data of file
        """
        response = self.s3.Object(bucket_name=self.bucket.name, key=os.path.join(conf.dirs.s3_videos, filename)).get()
        return response['Body'].read()

    def s3_vid_exists(self, filename):
        """check whether a file exists on s3. does not compare file sizes

        :param filename: name of file to check the s3 bucket for
        :return: boolean indicating if the file exists
        """
        try:
            self.s3.Object(self.bucket.name, os.path.join(conf.dirs.s3_videos, filename)).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # The object does not exist.
                return False
            else:
                # Something else has gone wrong.
                return False
        else:
            return True

    def s3_get_vid_size(self, filename):
        """get size of file on s3

        :param filename: name of file on s3
        :return: size in bytes
        """
        return self.client.head_object(Bucket=self.bucket.name, Key=os.path.join(conf.dirs.s3_videos, filename))["ContentLength"]
