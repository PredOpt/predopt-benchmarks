from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pytorch_lightning.callbacks import ModelCheckpoint 
from glob import glob
parent_dir=   log_dir+"lightning_logs/"
files = glob("lightning_logs/SPO30x30seed*")
for file_name in files:
	print(file_name)


# def datta_agg(parent_dir):

# 	version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

# 	walltimes = []
# 	steps = []
# 	regrets= []
# 	mses = []
# 	for logs in version_dirs:
# 	    event_accumulator = EventAccumulator(logs)
# 	    event_accumulator.Reload()

# 	    events = event_accumulator.Scalars("val_hammingloss")
# 	    walltimes.extend( [x.wall_time for x in events])
# 	    steps.extend([x.step for x in events])
# 	    regrets.extend([x.value for x in events])
# 	    events = event_accumulator.Scalars("val_mse")
# 	    mses.extend([x.value for x in events])

# 	df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_hammingloss": regrets,
# 	"val_mse": mses })
# 	df['model'] = modelname + args.loss
# 	df.to_csv(learning_curve_datafile,index=False)

# models = ['baseline','CachingPOlistwise','CachingPOpairwise','CachingPOpairwise_diff', 'CachingPOpointwise', 'DBB','DBBregret' ,
#  'FenchelYoung', 'IMLEregret','SPO']
# for model in models:
#     for img_string in ['12','18','24','30']:
#         try:
#         	dirname  = model +'{}x{}'.format(img_string, img_string)+'seed*'

#         except ValueError as err:
#           print   ('The actual error text is --> ', err, ' <--')
#           pass