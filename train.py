from sim.cartpole_nn_dc import CartPoleNNDCEnv as CartPoleEnv
from model import ActorModel, CriticModel, ActorCriticModel
import sys
from trainer.ddpg_trainer import DDPGTrainer
from trainer.ppo_trainer import PPOTrainer

def main():
	#trainer = DDPGTrainer(CartPoleEnv, ActorModel, CriticModel)
	trainer = PPOTrainer(CartPoleEnv, ActorCriticModel)
	try:
		trainer.train()
	except KeyboardInterrupt:
		trainer.env.close()
		if trainer.args.mode == 'train':
			trainer.agent.save_model('models', trainer.episode)
			print("Saved model: {}".format(trainer.episode))
	sys.exit(0)


if __name__ == '__main__':
	main()
