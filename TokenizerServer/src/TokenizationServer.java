import com.google.gson.Gson;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class TokenizationServer {
    public static void main(String[] args) {
        if (args.length > 1) {
            new TokenizationServer(Integer.valueOf(args[0]), Integer.valueOf(args[1]));
        } else {
            new TokenizationServer(6060, 6);
        }
    }

    public TokenizationServer(int port, int threadPoolSize) {
        ExecutorService pool = Executors.newFixedThreadPool(threadPoolSize);
        try {
            ServerSocket serverSocket = new ServerSocket(port);
            System.out.println("Started tokenizer server on the port " + port);
            while (true) {
                Socket clientSocket;
                try {
                    clientSocket = serverSocket.accept();
                } catch (IOException e) {
                    System.out.println("The server has unexpectedly stopped!");
                    e.printStackTrace();
                    break;
                }
                try {
                    pool.execute(new TokenizationTask(clientSocket));
                } catch (IOException e){
                    System.out.println("An error occurred while creating processor thread!");
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            System.out.println("An error occurred while creating server socket!");
            e.printStackTrace();
        }
    }

    private static class TokenizationTask implements Runnable {
        private final Socket clientSocket;
        private final BufferedWriter responseWriter;
        private final Tokenizer tokenizer;

        public TokenizationTask(Socket clientSocket) throws IOException {
            responseWriter = new BufferedWriter(new OutputStreamWriter(clientSocket.getOutputStream()));
            this.clientSocket = clientSocket;
            tokenizer = new Tokenizer();
        }

        @Override
        public void run() {
            BufferedReader requestReader = null;
            try {
                requestReader = new BufferedReader(new InputStreamReader(this.clientSocket.getInputStream(), "utf-8"));
                while (true) {
                    String response;
                    try {
                        String request = requestReader.readLine();
                        if (request == null) break;
                        response = new Gson().toJson(tokenizer.tokenize(request));
                    } catch (IOException ignored) {
                        break;
                    }
                    try {
                        this.responseWriter.write(response);
                        this.responseWriter.newLine();
                        this.responseWriter.flush();
                    } catch (IOException e) {
                        System.out.println("An error occurred while responding!");
                        e.printStackTrace();
                    }
                }
            } catch (IOException e) {
                System.out.println("An error occurred while creating request reader!");
                e.printStackTrace();
            } finally {
                if (requestReader != null) {
                    try {
                        requestReader.close();
                    } catch (IOException e) {
                        System.out.println("An error occurred while closing request reader!");
                        e.printStackTrace();
                    }
                }
                try {
                    this.clientSocket.close();
                } catch (IOException e) {
                    System.out.println("An error occurred while closing client socket!");
                    e.printStackTrace();
                }
                String hostName = this.clientSocket.getInetAddress().getHostName();
                int port = this.clientSocket.getPort();
                System.out.println("Session with client port:" + port + " host name:" + hostName + "is closed!");
            }
        }
    }
}