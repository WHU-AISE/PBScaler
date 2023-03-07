import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;


public class TestServiceStation {
    private WebDriver driver;
    private String baseUrl;
    @BeforeClass
    public void setUp() throws Exception {
        System.setProperty("webdriver.chrome.driver", "D:/Program/chromedriver_win32/chromedriver.exe");
        driver = new ChromeDriver();
        baseUrl = "http://10.141.212.24/";
        driver.manage().timeouts().implicitlyWait(30, TimeUnit.SECONDS);
    }
    @Test
    public void testStation() throws Exception{
        driver.get(baseUrl + "/");
        JavascriptExecutor js = (JavascriptExecutor) driver;
        js.executeScript("document.getElementById('station_update_id').value='shanghai'");
        js.executeScript("document.getElementById('station_update_name').value='shang hai'");

        driver.findElement(By.id("station_update_button")).click();
        Thread.sleep(1000);
 //       String statusStation = driver.findElement(By.id("login_result_msg")).getText();
//        if(statusSignIn ==null || statusSignIn.length() <= 0) {
//            System.out.println("Failed,Status of Sign In btn is NULL!");
//            driver.quit();
//        }else
//            System.out.println("Sign Up btn status:"+statusSignIn);
    }
    @Test (dependsOnMethods = {"testStation"})
    public void testQueryStation() throws Exception{
        driver.findElement(By.id("station_query_button")).click();
        Thread.sleep(1000);
        //gain Travel list
        List<WebElement> stationList = driver.findElements(By.xpath("//table[@id='query_station_list_table']/tbody/tr"));

        if (stationList.size() > 0)
            System.out.printf("Success to Query Station and Station list size is %d.%n",stationList.size());
        else
            System.out.println("Failed to Query Station or Station list size is 0");
        Assert.assertEquals(stationList.size() > 0,true);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
