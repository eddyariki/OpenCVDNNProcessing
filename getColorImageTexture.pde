void extractFaces(ArrayList<int[]> position, PImage gray) {
  for (int i=0; i<position.size(); i++) {
    {
      int x = position.get(i)[0];
      int y = position.get(i)[1];
      int w = position.get(i)[2]-x;
      int h = position.get(i)[3]-y;
      PImage grayFace = gray.get(x, y, w, h);
      faceGray.add(grayFace);

    }
  }
}
