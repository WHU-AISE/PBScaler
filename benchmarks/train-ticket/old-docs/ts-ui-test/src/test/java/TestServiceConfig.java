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

public class TestServiceConfig {
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
    public void testConfig() throws Exception{
        driver.get(baseUrl + "/");
        JavascriptExecutor js = (JavascriptExecutor) driver;
        js.executeScript("document.getElementById('config_update_name').value='DirectTicketAllocationProportion'");
        js.executeScript("document.getElementById('config_update_value').value='50%'");
        js.executeScript("document.getElementById('config_update_description').value='configtest'");

        driver.findElement(By.id("config_update_button")).click();
        Thread.sleep(1000);
//        String statusSignIn = driver.findElement(By.id("login_result_msg")).getText();
//        if(statusSignIn ==null || statusSignIn.length() <= 0) {
//            System.out.println("Failed,Status of Sign In btn is NULL!");
//            driver.quit();
//        }else
//            System.out.println("Sign Up btn status:"+statusSignIn);
    }
    @Test (dependsOnMethods = {"testConfig"})
    public void testQueryConfig() throws Exception{
        driver.findElement(By.id("config_query_button")).click();
        Thread.sleep(1000);
        //gain Travel list
        List<WebElement> configList = driver.findElements(By.xpath("//table[@id='query_config_list_table']/tbody/tr"));
        if (configList.size() > 0)
            System.out.printf("Success to Query Config and Config list size is %d.%n",configList.size());
        else
            System.out.println("Failed to Query Config or Config list size is 0");
        Assert.assertEquals(configList.size() > 0,true);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
