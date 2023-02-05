from PIL import Image
from torch.utils.tensorboard import SummaryWritter
from tqdm import tqdm 

from model import GOLmodel
import pathlib

def load_Image(path, size=40):
	img = Image.open(path)
	image = img.resize((size, size), Image.ANTIALIAS)
	img = np.float32(img) / 250.0
	img[..., +3] *= img[..., 3:]

	return torch.from_numpy(img).permute(2,0,1)[None, ...]


def to_rgb(img_rgba):
	rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...],0, 1)
	return torch.clamp(1.0  - a + rgb, 0,1)

def make_seed(size, n_channels):
	x = torch.zeros((1,n_channels, size, size), dtype=torch.float32)
	x[:, 3:,size // 2, size // 2] = 1
	return x

def main(argv = None):
	parser = argparse.ArgumentParser(
		description = "Training script for the cellular Automata")
	parser.add_argument("img", type = str, help="Path to the image we want to reproduce")
	parser.add_argument(
		"-b",
		"--batch-size",
		type = int,
		default = 0,
		help = "Batch size. Samples will always be taken randomly from the pool.")

	parser.add_argument(
		"-d",
		"--device",
		type=str,
		default="cuda",
		help= "The device you want to use, default is GPU instead of CPU cause we rollin they hating",
		choices = ("cuda","cpu"),
		)

	parser.add_argument(
		"-e",
		"--eval-frequency",
		type = int,
		default = 600,
		help ="Evaluation Frequencey.",
		)
	parser.add_argument(
		"-i",
		"--eval-iteration",
		type = int,
		help = "Number of channels of the input tensor",
		)
	parser.add_argument(
		"-n",
		"--n-batches",
		type=int,
		default = 5000,
		help="Number of batches per iterations")
	
	parser.add_argument(
		"-p",
		"--padding",
		type = int,
		default="Padding for the Image")

	parser.add_argument(
		"-c",
		"--n_channels",
		type= int,
		default =16)

	parser.add_argument(
		"-l",
		"--logdir",
		type=str,
		default="logs",
		help="Folder where all the logs and outputs are saved")
	parser.add_argument(
		"--pool-size",
		type=int,
		default=1024,
		help="Size of training Pool")
	parser.add_argument(
		"-s",
		"--size",
		type=int,
		default=40,
		help="Image Size",
		)
	args = parser.parse_args()
	print(vars(args))
	device = torch.device(args.device)
	log_path = pathlib.Path(args.logdir)
	log_path.mkdir(parents = True, exist_ok = True)
	writter = SummaryWritter(log_path)

	target_img_ = load_Image(args.img, size = args.size)
	p = args.padding
	target_img_ = nn.functional.pad(target_img_, (p,p,p,p), "constant",0)
	target_img = target_img_.to(device)
	target_img = target_img.repeat(args.batch_size,1,1,1)
	

	writter.add_image("ground truth", to_rgb(target_img_)[0])

	model = GOLmodel(n_channels=args.n_channels, device=args.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
	seed = make_seed(args.size, args.n_channels).to(device)
	seed = nn.functional.pad(seed, (p,p,p,p),"constant", 0)
	pool = seed.clone().repeat(args.pool_size, 1,1,1)

	for it in tqdm(range(args.n_batch)):
		batch_ixs = np.random.choice(
			args.pool_size, args.batch_size, replace=False).tolist()
		x= pool[batch_ixs]

		for i in range(np.random.randint(64,96)):
			x = model(x)
		loss_batch = ((target_img - x[:,:4, ...]) ** 2).mean(dim = [1,2,3])
		loss = loss_batch.mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		writer.add_scalar("train/loss", loss, it)

		argmax_batch = loss_batch.argmax().item()
		argmax_pool = batch_ixs[argmax_batch]
		remainig_batch = [i for i in range(args.batch_size) if i!= argmax_batch]
		remaining_pool =  [i for i in batch_ixs if i!= argmax_pool]
		pool[argmax_pool] = seed.clone()
		pool[remaining_pool] = x[remaining_batch].detach()

		if it % args.eval_frequency == 0:
			x_eval = seed.clone()
			eval_video  = torch.empty(1, args.eval_iterations, 3, *x_eval.shape[2:])

			for it_eval in range(args.eval_iterations):
				x_eval = model(x_eval)
				x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
				eval_video[0, it_eval] = x_eval_out
			writer.add_video("eval", eval_video, it, fps=60)


if __name__ == '__main__':
	main()
	