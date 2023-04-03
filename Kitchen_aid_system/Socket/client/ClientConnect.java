import java.net.*;
import java.io.*;
import java.util.Scanner;

public class ClientConnect {

    private Socket socket = null;
    private FileOutputStream fos = null;
    private DataInputStream din = null;
    private PrintStream pout = null;
    private Scanner scan = null;

    public void Connect(InetAddress address, int port) throws IOException
    {
        System.out.println("Initializing Client");
        socket = new Socket(address, port);
        scan = new Scanner(System.in);
        din = new DataInputStream(socket.getInputStream());
        pout = new PrintStream(socket.getOutputStream());
    }

    public void send(String msg) throws IOException
    {
        pout.print(msg);
        pout.flush();
    }

    public String recv() throws IOException
    {
        byte[] bytes = new byte[1024];
        din.read(bytes);
        String reply = new String(bytes, "UTF-8");
        System.out.println("Inside recv(): ");
        return reply;
    }

    public void closeConnections() throws IOException
    {
        // Clean up when a connection is ended
        socket.close();
        din.close();
        pout.close();
        scan.close();
    }

    public void chat() throws IOException
    {
        String response = "s";

        System.out.println("Initiating Chat Sequence");
        while(!response.equals("QUIT")){
            System.out.print("Client: ");
            String message = scan.nextLine();
            send(message);
            if(message.equals("QUIT"))
                break;
            response = recv();
            System.out.println("Server: " + response);
        }
        closeConnections();
    }

    // Request a specific file from the server
    public void getFile(String filename)
    {
        System.out.println("Requested File: "+filename);
        try {
            File file = new File(filename);
            // Create new file if it does not exist
            // Then request the file from server
            if(!file.exists()){
                file.createNewFile();
                System.out.println("Created New File: "+filename);
            }
            fos = new FileOutputStream(file);
            send(filename);

            // Get content in bytes and write to a file
            // while((counter = din.read(buffer, 0, buffer.length)) > 0)
            int counter = 0;
            byte[] buffer = new byte[8192];
            while((counter = din.read(buffer, 0, buffer.length)) > 0)
            {
                fos.write(buffer, 0, counter);
                System.out.println(counter);
                System.out.println(buffer.length);
            }

            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    public static void main(String[] args) throws IOException
    {
        InetAddress ip = null;
        String input = null;
        String receive = "s";
        ip = InetAddress.getByName("192.168.190.1");

        ClientConnect connect = new ClientConnect();

        connect.Connect(ip, 8000);
        connect.send("test for connect");

        Scanner user_input = new Scanner(System.in);
        while(true)
        {
            receive = connect.recv();
            System.out.println(receive);
            if (input == null)
            {
                input = user_input.nextLine();
            }

            if (input.contains("chat"))
            {
                connect.send("chat");
                System.out.println("send chat signal");
                connect.chat();
                input = null;
            }
            else if (input.contains("file"))
            {
                connect.send("file");
                // connect.send("light.png");
                System.out.println("send file transfer signal");
                connect.getFile("light.png");
                //connect.send("got file");
                input = null;
            }
            else if (input.contains("reset"))
            {
                connect.send("reset");
                System.out.println("send reset signal");
                input = null;
            }
            else if (input.contains("exit"))
            {
                connect.send("exit");
                System.out.println("send exit signal");

                connect.closeConnections();
                input = null;
            }
            else
            {
                connect.send(input);
                input = null;
            }
/*
            if (receive.contains("true"))
            {
                System.out.println(receive);
                connect.send("Receive data");
                connect.closeConnections();
            }
            else
            {
                System.out.println(receive);
                connect.send("Not True");
            }
*/
        }

    }

}
