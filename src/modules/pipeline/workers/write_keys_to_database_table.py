# ============ Base imports ======================
import os
import json
from collections import Iterable
# ====== External package imports ================
import numpy as np
# ====== Internal package imports ================
from src.modules.pipeline.workers.pipeline_worker import PipelineWorker
from src.modules.data.database_io import DatabaseIO
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class WriteKeysToDatabaseTable(PipelineWorker):
    def initialize(self, keys, keys_headers, buffer_size, schemas, tables, field_separator, additional_data, **kwargs):
        self.keys = keys
        self.keys_headers = keys_headers
        self.buffer_size = buffer_size
        self.schemas = schemas
        self.tables = tables
        self.field_separator = field_separator
        self.additional_data = additional_data

    def startup(self):
        self.dbio = DatabaseIO()
        self.buffer = []
        self.buffer_fill = 0
        self.written_headers = [False] * len(self.keys)

    def run(self, item, *args, **kwargs):
        # for each key, write to its file
        self.buffer.append(item)
        self.buffer_fill += 1
        if self.buffer_fill == self.buffer_size:
            self.logger.info("Buffer full, writing to DB")
            self.copy_buffer_to_db()
            for item in self.buffer:
                self.done_with_item(item)
            self.buffer = []
            self.buffer_fill = 0

    def shutdown(self):
        if self.buffer_fill > 0:
            self.logger.info("Flushing buffer before shutdown")
            self.copy_buffer_to_db()
            for item in self.buffer:
                self.done_with_item(item)
            self.buffer = []
            self.buffer_fill = 0

    def copy_buffer_to_db(self):
        for i in range(len(self.keys)):
            key = self.keys[i]
            self.logger.debug(f"writing key:{key}")
            data = f'"model_number"{self.field_separator}"video_id"{self.field_separator}"video_file_name"{self.field_separator}"frame_number"{self.field_separator}"' + ('"' + self.field_separator + '"').join(self.buffer[0][self.keys_headers[i]]) + '"'
            if self.additional_data is not None:
                if self.additional_data[i] not in self.buffer[0].keys():
                    self.logger.error(f"Key {self.additional_data[i]} not found in item, not writing to database")
                    continue
                data += f'{self.field_separator}\"{self.additional_data[i]}\"'
            for item in self.buffer:
                prefix = f"\"{self.model_number}\"{self.field_separator}\"{item['video_info']['id']}\"{self.field_separator}\"{item['video_info']['file_name']}\"{self.field_separator}\"{item['frame_number']}\""
                if self.additional_data is not None:
                    string = self.make_string(prefix, item[key], len(item[self.keys_headers[i]]), additional_data=item[self.additional_data[i]])
                else:
                    string = self.make_string(prefix, item[key], len(item[self.keys_headers[i]]))
                if string is not None:
                    data += "\n" + string
            self.dbio.copy_string_to_table(self.schemas[i], self.tables[i], data, "\t")

    def make_string(self, prefix, data, length, additional_data=None):
        if data is None:
            if additional_data is not None:
                return prefix + (self.field_separator * (length+1))
            else:
                return prefix + (self.field_separator * length)
        # if the data are not a list, just convert to string
        if not isinstance(data, Iterable):
            if additional_data is not None:
                return f"{prefix}{self.field_separator} {str(data)}{self.field_separator} {additional_data}"
            else:
                return f"{prefix}{self.field_separator} {str(data)}"
        datal = [datum for datum in data]
        if len(datal) == 0:
            if additional_data is not None:
                return prefix + (self.field_separator * (length+1))
            else:
                return prefix + (self.field_separator * length)
        if isinstance(datal[0], Iterable):
            nl = " \n"
            if additional_data is not None:
                return  nl.join([f"{prefix}{self.field_separator}{self.field_separator.join([str(datum2) for datum2 in datum])}{self.field_separator}{add_datum} " for datum, add_datum in zip(datal, additional_data)])
            else:
                return nl.join([f"{prefix}{self.field_separator}{self.field_separator.join([str(datum2) for datum2 in datum])} " for datum in datal])
        else:
            if additional_data is not None:
                return f"{prefix}{self.field_separator}{self.field_separator.join([str(datum) for datum in datal])}"
            else:
                return f"{prefix}{self.field_separator}{self.field_separator.join([str(datum) for datum in datal])}{self.field_separator}{additional_data}"
