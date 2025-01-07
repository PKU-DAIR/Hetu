import threading
from hetu.rpc import heturpc_pb2
from .const import *

def key_value_store_server(cls):
    cls.json_dicts = {}
    cls.dict_locks = {}
    cls.kv_store_global_lock = threading.Lock()

    def parse_key(self, combined_key):
        parts = combined_key.split(DELIMITER, 1)
        if len(parts) != 2:
            raise ValueError(f"Key must be in the format '<dict_name>{DELIMITER}<key>'")
        return parts[0], parts[1]

    def get_dict_lock(self, dict_name):
        with self.kv_store_global_lock:
            if dict_name not in self.dict_locks:
                self.json_dicts[dict_name] = {}
                self.dict_locks[dict_name] = threading.Condition()
            return self.dict_locks[dict_name]

    def PutJson(self, request, context):
        dict_name, key = self.parse_key(request.key)
        with self.get_dict_lock(dict_name):
            self.json_dicts[dict_name][key] = request.value
            print(f"Server: PutJson in dict '{dict_name}' key '{key}'")
            self.dict_locks[dict_name].notify_all()
        return heturpc_pb2.PutJsonReply(status=1)

    def GetJson(self, request, context):
        dict_name, key = self.parse_key(request.key)
        with self.get_dict_lock(dict_name):
            while key not in self.json_dicts.get(dict_name, {}):
                self.dict_locks[dict_name].wait()
            value = self.json_dicts[dict_name][key]
            print(f"Server: GetJson from dict '{dict_name}' key '{key}'")
            return heturpc_pb2.GetJsonReply(value=value)

    def RemoveJson(self, request, context):
        dict_name, key = self.parse_key(request.key)
        with self.get_dict_lock(dict_name):
            if key in self.json_dicts.get(dict_name, {}):
                del self.json_dicts[dict_name][key]
                print(f"Server: RemoveJson from dict '{dict_name}' key '{key}' removed")
                message = "removed"
            else:
                # print(f"Server: RemoveJson key '{key}' not found in dict '{dict_name}'")
                message = "key not found"
        return heturpc_pb2.RemoveJsonReply(message=message)

    cls.parse_key = parse_key
    cls.get_dict_lock = get_dict_lock
    cls.PutJson = PutJson
    cls.GetJson = GetJson
    cls.RemoveJson = RemoveJson

    return cls

