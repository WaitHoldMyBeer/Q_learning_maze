import pygame
import pandas as pd
from thirtytwostatepygame import Environment, QLearningModel, Screen

pygame.init()
        
def main():
    rat_data = pd.read_csv("main_project/modules/rat1_data.csv")
    print(rat_data.head())
    screen = Screen()
    maze_width = screen.width//screen.cell_size
    maze_height = (screen.height-50)//screen.cell_size
    environment = Environment(maze_width, maze_height, rat_data.loc[0]["State"], start = [0,2])
    q_model = QLearningModel()
    running = True
    index = 0
    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_1:
            prediction = q_model.choose_action(environment.state)
            screen.display_text.update({"predicted state": f"{prediction}"})
            q_model.step(environment)
          if event.key == pygame.K_2:
              for _ in range(10):
                prediction = q_model.choose_action(environment.state)
                screen.display_text.update({"predicted state": f"{prediction}"})
                q_model.step(environment)
          if event.key == pygame.K_r:
            prediction = q_model.choose_action(environment.state)
            print("tabled_start =", rat_data.iloc[index]['Start'])
            screen.display_text.update({"predicted state": f"{prediction}"})
            if (environment.state != rat_data.iloc[index]["State"]):
                print("Error, states did not match", environment.state, rat_data.iloc[index]["State"])
            q_model.step(environment, rat_data.iloc[index]['Action'], start=rat_data.iloc[index+1]['Start'])
            
            prediction = q_model.choose_action(environment.state) # this is the predicted state
            index += 1
          # print(model.choose_action(STATES_TO_COORDINATES.index((agent.y, agent.x))))
          if event.key == pygame.K_UP:
            prediction = q_model.choose_action(environment.state)
            screen.display_text.update({"predicted state": f"{prediction}"})
            q_model.step(environment, 0)
          elif event.key == pygame.K_DOWN:
            prediction = q_model.choose_action(environment.state)
            screen.display_text.update({"predicted state": f"{prediction}"})
            q_model.step(environment, 3)
          elif event.key == pygame.K_LEFT:
            prediction = q_model.choose_action(environment.state)
            screen.display_text.update({"predicted state": f"{prediction}"})
            q_model.step(environment, 1)
          elif event.key == pygame.K_RIGHT:
            prediction = q_model.choose_action(environment.state)
            screen.display_text.update({"predicted state": f"{prediction}"})
            q_model.step(environment, 2)
        screen.draw(environment)
        pygame.display.flip()
    screen.run(environment, q_model)
    pygame.quit()
    


if __name__ == "__main__":
    main()