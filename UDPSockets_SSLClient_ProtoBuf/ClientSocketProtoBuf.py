import ssl_client_pb2 as TodoList

my_list = TodoList.SSL_DetectionRobot()
my_list.confidence = 99
my_list.robot_id = 1
my_list.x = 1
my_list.y = 1
my_list.orientation = 1
my_list.pixel_x = 1
my_list.pixel_y = 1
my_list.height  = 125

# first_item = my_list.todos.add()
# first_item.state = TodoList.TaskState.Value("TASK_DONE")
# first_item.task = "Test ProtoBuf for Python"
# first_item.due_date = "31.10.2019"

print(my_list)