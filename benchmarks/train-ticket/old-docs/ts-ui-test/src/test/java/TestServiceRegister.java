import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.concurrent.TimeUnit;

/**
 * Created by ZDH on 2017/7/21.
 */
public class TestServiceRegister {
    private WebDriver driver;
    private String baseUrl;
    public static void login(WebDriver driver,String username,String password){
        driver.findElement(By.id("flow_one_page")).click();
        driver.findElement(By.id("flow_preserve_login_email")).clear();
        driver.findElement(By.id("flow_preserve_login_email")).sendKeys(username);
        driver.findElement(By.id("flow_preserve_login_password")).clear();
        driver.findElement(By.id("flow_preserve_login_password")).sendKeys(password);
        driver.findElement(By.id("flow_preserve_login_button")).click();
    }
    @BeforeClass
    public void setUp() throws Exception {
        System.setProperty("webdriver.chrome.driver", "D:/Program/chromedriver_win32/chromedriver.exe");
        driver = new ChromeDriver();
        baseUrl = "http://10.141.212.24/";
        driver.manage().timeouts().implicitlyWait(30, TimeUnit.SECONDS);
    }
    @DataProvider(name="user")
    public Object[][] Users(){
        return new Object[][]{
                {"2daihongok@163.com","DefaultPassword"},
        };
    }
    @Test (dataProvider="user")
    public void testRegister(String username,String password) throws Exception{
        driver.get(baseUrl + "/");

        driver.findElement(By.id("register_email")).clear();
        driver.findElement(By.id("register_email")).sendKeys(username);
        driver.findElement(By.id("register_password")).clear();
        driver.findElement(By.id("register_password")).sendKeys(password);

        driver.findElement(By.id("register_button")).click();
        Thread.sleep(1000);

        String statusSignUp = driver.findElement(By.id("register_result_msg")).getText();
        if ("".equals(statusSignUp))
            System.out.println("Failed,Status of Sign Up btn is NULL!");
        else
            System.out.println("Sign Up btn status:"+statusSignUp);
        Assert.assertEquals(statusSignUp.startsWith("Success"),true);
    }
    @Test (dependsOnMethods = {"testRegister"},dataProvider="user")
    public void testRegisterLogin(String username,String password) throws Exception{
        //call function login
        login(driver,username,password);
        Thread.sleep(1000);

        //get login status
        String statusLogin = driver.findElement(By.id("flow_preserve_login_msg")).getText();
        if(statusLogin.startsWith("Success")) {
            System.out.println("Login status:"+statusLogin);
            driver.findElement(By.id("microservice_page")).click();
        }
        else if("".equals(statusLogin))
            System.out.println("False,Failed to login! StatusLogin is NULL");
        else
            System.out.println("Failed to login!" + "Wrong login Id or password!");

        Assert.assertEquals(statusLogin.startsWith("Success"),true);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
