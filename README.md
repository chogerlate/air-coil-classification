# air-coil-classification
Predicting air coil condition from image. 

# How to use this repository 
1. Clone this repo.
2. Load modle weight for air coil classification model, currently support only eva02 model which is the best model.

    |   Model   |   Link   |
    | --------- | -------- |
    |   EVA02   | https://drive.google.com/file/d/1W7O3aB4IQ6feYyW6LKtRy2b6KN5Dyn5f/view?usp=drive_link |

3. Run Docker Command

    docker build
    ```
    docker build -t aircoilmodel
    ```

    docker run 
    ```
    docker run --name aircoiltest -d -p 8080:80 aircoilmodel
    ```