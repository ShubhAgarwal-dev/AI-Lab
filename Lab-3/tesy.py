from timeit import default_timer

initial_clock = default_timer()

from  file_reader import array_converter
vals = array_converter(r"D:\Projects\AI Lab\Lab-3\euc_100")
end_clock = default_timer()
print(end_clock - initial_clock)
