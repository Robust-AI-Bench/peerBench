
PWD=$(pwd) ;  
REPO=$(basename $PWD)
NAME=$REPO-test
TEST_PATH=/app/$REPO/test.py
TEST_CMD="d test"
./run/stop.sh $NAME
docker run -d --name $NAME -v $PWD:/app $REPO
docker exec -it $NAME bash -d "$TEST_CMD"
./run/stop.sh $NAME
